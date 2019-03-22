import os
import math
import base64
import itertools
import sys
import random
import binascii
import argparse
import bcrypt
import time
import string
import urllib.request
import requests
import functools
import logging
import json
import peewee as pw
import flask
import gunicorn.app.wsgiapp
import jinja2

nginx_conf = 'nginx.conf'
user_registration_conf = 'user_registration.json'
incomplete_task_threshold = dict(vote = 4)

class User(pw.Model):
	id = pw.PrimaryKeyField()
	display = pw.TextField()
	email = pw.TextField(default = '')
	email_normalized = pw.TextField(default = '')
	password = pw.TextField(default = lambda: binascii.hexlify(os.urandom(8)).decode('ascii'), null = True)
	hashed = pw.TextField(null = True)
	role = pw.TextField(default = 'task/vote')

	def hash_password(self):
		self.hashed = bcrypt.hashpw(self.password.encode('utf-8'), bcrypt.gensalt())
		self.password = None

	def check_password(self, user):
		return bcrypt.checkpw(self.password.encode('utf-8'), user.hashed.encode('utf-8'))

	def generate_token(self):
		return base64.b64encode(json.dumps(dict(id = self.id, display = self.display, role = self.role, password = self.password)).encode('ascii')).decode('ascii').rstrip('=')

	def parse_token(self, user_token):
		parsed = json.loads(base64.b64decode(user_token + '===').decode('ascii'))
		self.id, self.display, self.role, self.password = map(parsed.get, ['id', 'display', 'role', 'password'])

	def is_admin(self):
		return 'admin' in self.role

	def send_registration_email(self, user_token, registration_email_subject, registration_email_body, sender_email, sender_name, endpoint, headers, data, website_http_location, debug_user_registration_email_file_path):
		payload = string.Template(data).substitute(registration_email_subject = registration_email_subject, registration_email_body = string.Template(registration_email_body).substitute(user_token = user_token, website_http_location = website_http_location), sender_email = sender_email, sender_name = sender_name, recipient = self.email)
		if debug_user_registration_email_file_path is not None:
			open(debug_user_registration_email_file_path, 'w').write(payload)
		else:
			requests.post(endpoint, headers = headers, data = payload.encode('utf-8'))

	@staticmethod
	def normalize_email(email):
		splitted = email.split('@')
		return splitted[0].split('+')[0].replace('.', '') + '@' + (splitted[1] if len(splitted) >= 2 else '')

class Station(pw.Model):
	id = pw.PrimaryKeyField()
	station_number = pw.IntegerField()
	region_number = pw.IntegerField()
	election_number = pw.IntegerField()
	station_address = pw.TextField()
	timezone_offset = pw.IntegerField()
	station_interval_start = pw.IntegerField()
	station_interval_end = pw.IntegerField()

	@property
	def station_id(self):
		return '{:06d}.{:02d}.{:04d}'.format(self.election_number, self.region_number, self.station_number)

class Clip(pw.Model):
	id = pw.PrimaryKeyField()
	station = pw.ForeignKeyField(Station, backref = 'clips')
	video = pw.TextField(null = True)
	thumbnail = pw.TextField(null = True)
	clip_interval_start = pw.IntegerField(null = True)
	clip_interval_end = pw.IntegerField(null = True)
	meta = pw.TextField(default = '')
	task = pw.TextField(default = '')
	camera_id = pw.TextField(default = '')
	csrf = pw.IntegerField(default = lambda: random.randint(0, 1e9))
	gold = pw.BooleanField(default = False)

class Event(pw.Model):
	creator = pw.ForeignKeyField(User, null = True)
	timestamp = pw.IntegerField(default = lambda : int(time.time()))
	clip = pw.ForeignKeyField(Clip, null = True, backref = 'events')
	station = pw.ForeignKeyField(Station, null = True, backref = 'events')
	value = pw.TextField()
	offset = pw.DoubleField(default = 0.0)
	type = pw.TextField()

def user_must_be_active(admin = False, fail = True):
	def wrap(method):
		@functools.wraps(method)
		def wrapped(*args, **kwargs):
			try:
				user_stub = User()
				user_stub.parse_token(flask.request.cookies['user_token'])
				user = User.get_or_none(User.id == user_stub.id)
				if user is None or not user_stub.check_password(user) or (admin and not user.is_admin()):
					raise Exception()
			except:
				user = None
				if fail:
					flask.abort(401)
			return method(user = user, *args, **kwargs)
		return wrapped
	return wrap

def estimate_clip(clip):
	group_key = lambda ev: ev.creator_id
	num_votes = [len(list(g)) for k, g in itertools.groupby(sorted(filter(lambda ev: ev.type == 'vote', clip.events), key = group_key), group_key)]
	estimate = ([json.loads(ev.value) for ev in clip.events if ev.type == 'turnout_estimate'] + [None])[0]
	return dict(final = estimate, average = int(math.ceil(sum(num_votes) / float(len(num_votes)))) if len(num_votes) > 0 else None, completed = len(num_votes) >= incomplete_task_threshold['vote'] )

def estimate_station(station, clip_turnout):
	clip_turnout_ = [clip_turnout[clip.id] for clip in station.clips]
	official = ([json.loads(ev.value) for ev in station.events if ev.type == 'turnout_official'] + [{}])[0]
	progress = sum(1 if t['completed'] else 0 for t in clip_turnout_) / float(len(clip_turnout_))
	comment = ' || '.join(t['final']['comment'] for t in clip_turnout_ if (t['final'] or {}).get('comment') not in ['', None])

	def interval_turnout(hours_begin = None, hours_end = None, timestamp_begin = None, timestamp_end = None, hours_baseline = 8, normalize = True):
		interval_start, interval_end = ((station.station_interval_start - hours_baseline * 60 * 60) + hours_begin * 60 * 60, (station.station_interval_start - hours_baseline * 60 * 60) + hours_end * 60 * 60) if hours_begin is not None and hours_end is not None else (timestamp_begin, timestamp_end)
		clip_turnout_in = [clip_turnout[clip.id] for clip in station.clips if (interval_start <= clip.clip_interval_start and clip.clip_interval_end <= interval_end) or (clip.clip_interval_start <= interval_end and clip.clip_interval_end >= interval_end)]
		numer = sum(t['final']['count'] for t in clip_turnout_in) if all(t['final'] is not None for t in clip_turnout_in) else None
		denom = official.get('voters_registered')
		return numer if not normalize else (numer / denom if numer is not None and denom is not None else None)

	estimate = {'final' : interval_turnout(timestamp_begin = station.station_interval_start, timestamp_end = station.station_interval_end, normalize = False), '10h' : interval_turnout(8, 10), '12h' : interval_turnout(10, 12), '15h' : interval_turnout(12, 15), '18h' : interval_turnout(15, 18), '20h' : interval_turnout(18, 20)}
	return dict(estimate = estimate, official = official, comment = comment, progress = progress)

def stats_get():
	stations, clips = list(Station.select().prefetch(Event).prefetch(Clip)), list(Clip.select().where(Clip.task == 'vote').prefetch(Event).prefetch(Station))
	clip_turnout = {clip.id : estimate_clip(clip) for clip in clips}
	station_turnout = {station.id : estimate_station(station, clip_turnout) for station in stations}

	groupby = lambda l, key: [(k, list(g)) for k, g in itertools.groupby(sorted(l, key = key), key = key)]
	by_station = lambda stations: [('УИК #{station.station_number}'.format(station = station), station.station_number) for station in sorted(stations, key = lambda s: s.station_number)]
	by_region = lambda stations: [(g[0].station_address.split(',')[0], k, by_station(g)) for k, g in groupby(stations, key = lambda s: s.region_number)]
	by_election = lambda stations: [('Выборы {k}'.format(k = k), k, by_region(g)) for k, g in groupby(stations, key = lambda s: s.election_number)]

	return flask.Response(response = json.dumps(dict(

		stations = [dict(id = station.id, station_id = station.station_id, station_number = station.station_number, region_number = station.region_number, election_number = station.election_number, station_address = station.station_address, timezone_offset = station.timezone_offset, station_interval_start = station.station_interval_start, station_interval_end = station.station_interval_end, turnout = station_turnout[station.id], clips = ','.join(str(clip.id) for clip in station.clips)) for station in Station.select().order_by(Station.election_number, Station.region_number, Station.station_number).prefetch(Clip)],

		clips = [dict(id = clip.id, thumbnail = clip.thumbnail, video = clip.video, station_id = clip.station_id, clip_interval_start = clip.clip_interval_start, clip_interval_end = clip.clip_interval_end, turnout = clip_turnout[clip.id]) for clip in clips],
		num_users = User.select().count(),
		num_stations_labeled = sum(1 for turnout in station_turnout.values() if turnout['estimate'].get('final') is not None),
		num_stations = Station._meta.database.execute_sql(
			'SELECT IFNULL(COUNT(DISTINCT c.station_id), 0) '
			'FROM Clip c '
			'WHERE c.task == "vote"'
		).fetchone()[0],
		num_seconds = Clip._meta.database.execute_sql(
			'SELECT IFNULL(SUM(IFNULL(c.clip_interval_end, 0) - IFNULL(c.clip_interval_start, 0)), 0)'
			'FROM Clip c '
			'WHERE c.task == "vote"'
		).fetchone()[0],
		num_seconds_labeled = Clip._meta.database.execute_sql(
			'SELECT IFNULL(SUM(IFNULL(c.clip_interval_end, 0) - IFNULL(c.clip_interval_start, 0)), 0)'
			'FROM Clip c '
			'INNER JOIN Event ev ON ev.clip_id == c.id AND ev.type == "vote" '
			'WHERE c.task == "vote"'
		).fetchone()[0],
		events = [dict(id = ev.id, timestamp = ev.timestamp, value = ev.value, station_id = ev.station_id if ev.station is not None else ev.clip.station_id if ev.clip is not None else None) for ev in Event.select().where(Event.type == 'bookmark').prefetch(Clip)],
		users = list(User.raw(
			'SELECT u.id, u.display, IFNULL(SUM(e.type == "vote"), 0) as num_votes, SUM(IFNULL(c.clip_interval_end - c.clip_interval_start, 0)) as num_seconds, IFNULL(COUNT(DISTINCT c.station_id), 0) as num_stations, IFNULL(SUM(e.type == "note"), 0) as num_notes, IFNULL(GROUP_CONCAT(DISTINCT e.clip_id), "") as clips, IFNULL(COUNT(DISTINCT e.clip_id), 0) as num_clips, IFNULL(GROUP_CONCAT(DISTINCT eb.id), "") as bookmarks '
			'FROM User u '
			'LEFT OUTER JOIN Event e ON e.creator_id = u.id AND e.type != "bookmark" '
			'LEFT OUTER JOIN Event eb ON eb.creator_id = u.id AND eb.type == "bookmark" '
			'LEFT OUTER JOIN Clip c ON c.id = e.clip_id '
			'GROUP BY u.id, u.display '
			'ORDER BY num_votes DESC'
		).dicts()),
		task_selector_options = by_election(stations)

	), ensure_ascii = False, indent = 2), status = 200, mimetype = 'application/json')

@user_must_be_active()
def task_get(task_type, user, active_set = 20):
	clip = None
	if not (user is None or not (user.is_admin() or ('task/' + task_type) in user.role)):
		incomplete_tasks = list(Clip.raw(
			'SELECT c.*,  IFNULL(COUNT(DISTINCT ev.creator_id), 0) as num_completed, IFNULL(MAX(ev.creator_id == ?), 0) as is_completed '
			'FROM Clip c '
			'JOIN Station s ON c.station_id = s.id '
			'LEFT OUTER JOIN Event ev ON ev.clip_id == c.id AND ev.type == "vote" '
			'WHERE c.task == ? '
			'GROUP BY c.id, c.station_id, c.video, c.thumbnail, c.clip_interval_start, c.clip_interval_end, s.election_number, s.region_number, s.station_number '
			'HAVING is_completed == 0 AND num_completed < ? '
			'ORDER BY c.gold DESC, num_completed DESC, s.election_number ASC, s.region_number ASC, s.station_number ASC, c.clip_interval_start ASC, c.clip_interval_end ASC '
			'LIMIT ?',
			user.id, task_type, incomplete_task_threshold[task_type], active_set
		)) or [None]
		clip = random.choice(incomplete_tasks[:active_set])
	return flask.Response(response = json.dumps(dict(id = clip.id, task = clip.task, thumbnail = clip.thumbnail, video = clip.video, clip_interval_start = clip.clip_interval_start, clip_interval_end = clip.clip_interval_end, csrf = clip.csrf, station = dict(station_number = clip.station.station_number, station_address = clip.station.station_address, timezone_offset = clip.station.timezone_offset, election_number = clip.station.election_number)) if clip is not None else None, ensure_ascii = False), status = 200, mimetype = 'application/json')

@user_must_be_active()
def events_post(clip_id, user):
	if len(flask.request.get_json()) > 0 and Clip.get(Clip.id == clip_id).csrf == int(flask.request.args.get('csrf', 0)):
		Event.insert_many([dict(creator = user, clip = ev.get('clip'), value = ev.get('value', ''), offset = ev['offset'], type = ev['type']) for ev in flask.request.get_json() if ev.get('clip') == clip_id]).execute()
	return flask.jsonify(success = True)

def user_post():
	settings = json.load(open(user_registration_conf))
	display = random.choice(settings['nicknames']) + '_' + str(random.randint(100, 1000))

	email = flask.request.data.decode('utf-8')
	email_normalized = User.normalize_email(email)

	if User.get_or_none(User.email_normalized == email_normalized) is None:
		user = User.create(email = email, email_normalized = email_normalized, display = display)
		user_token = user.generate_token()
		user.hash_password()
		user.save()
		user.send_registration_email(user_token = user_token, registration_email_subject = settings['registration_email_subject'], registration_email_body = settings['registration_email_body'], sender_email = settings['sender_email'], sender_name = settings['sender_name'], website_http_location = settings['website_http_location'], debug_user_registration_email_file_path = settings.get('debug_user_registration_email_file_path'), **settings['http'])

	return flask.jsonify(success = True)

def init_db(db_path):
	db = pw.SqliteDatabase(db_path, autocommit = False)
	for model in [User, Station, Clip, Event]:
		model._meta.database = db
	return db

def import_(db_path, clips_path, stations_path, turnout = False, gold = 0):
	init_db(db_path)
	json_load = lambda uri: json.loads((open(uri, 'rb') if not uri.startswith('http') else urllib.request.urlopen(uri)).read().decode('utf-8'))
	if clips_path is not None and turnout is False:
		clips = json_load(clips_path)
		Clip.insert_many([dict(clip_interval_start = c.get('clip_interval_start'), clip_interval_end = c.get('clip_interval_end'), video = c.get('video'), thumbnail = c.get('thumbnail'), meta = c.get('meta', ''), station_id = Station.get_or_create(station_number = c['station_number'], region_number = c['region_number'], election_number = c['election_number'], defaults = dict(station_address = c['station_address'], timezone_offset = c['timezone_offset'], station_interval_start = c['station_interval_start'], station_interval_end = c['station_interval_end']))[0].id, task = c.get('task', ''), camera_id = c.get('camera_id' ''), gold = c.get('gold', random.random() < float(gold) / len(clips))) for c in clips]).execute()
	elif clips_path is not None and turnout is True:
		clips = json_load(clips_path)
		turnout_estimate = {clip_estimate['id'] : clip_estimate for clip_estimate in clips}
		Event.insert_many([dict(clip = clip, type = 'turnout_estimate', value = json.dumps(turnout_estimate[clip.id], ensure_ascii = False) ) for clip in list(Clip.select().prefetch(Event)) if not any(ev.type == 'turnout_estimate' for ev in clip.events) and clip.id in turnout_estimate]).execute()
	elif stations_path is not None and turnout:
		stations = json_load(stations_path)
		turnout_official = {station['station_id'] : station for station in stations}
		Event.insert_many([dict(station = station, type = 'turnout_official', value = json.dumps({'10h' : turnout_official[station.station_id]['turnout_10h'], '12h' : turnout_official[station.station_id]['turnout_12h'], '15h' : turnout_official[station.station_id]['turnout_15h'], '18h' : turnout_official[station.station_id]['turnout_18h'], '20h' : turnout_official[station.station_id]['ballots_given_at_station_on_election_day'] / float(turnout_official[station.station_id]['voters_registered']), 'final' : turnout_official[station.station_id]['ballots_given_at_station_on_election_day']})) for station in list(Station.select().prefetch(Event)) if not any(ev.type == 'turnout_official' for ev in station.events)]).execute()

def export_(stations_path, db_path):
	init_db(db_path)
	json.dump([dict(id = station.id, station_number = station.station_number, region_number = station.region_number, election_number = station.election_number, station_address = station.station_address, timezone_offset = station.timezone_offset, station_interval_start = station.station_interval_start, station_interval_end = station.station_interval_end, clips = [dict(camera_id = clip.camera_id, thumbnail = clip.thumbnail, task = clip.task, events = [dict(id = ev.id, timestamp = ev.timestamp, type = ev.type, offset = ev.offset, value = ev.value, creator = ev.creator.display, clip = ev.clip_id, station = ev.station_id) for ev in clip.events], completed = estimate_clip(clip)['completed']) for clip in station.clips]) for station in Station.select().prefetch(Clip).prefetch(Event).prefetch(User)], open(stations_path, 'w'), ensure_ascii = False, indent = 2, sort_keys = True)

def serve(db_path, log_sql, gunicorn_args):
	db = init_db(db_path)
	if log_sql:
		logger = logging.getLogger('peewee')
		logger.addHandler(logging.StreamHandler())
		logger.setLevel(logging.DEBUG)

	api = flask.Flask(__name__, static_url_path = '')
	api.route('/stats', methods = ['GET'])(stats_get)
	api.route('/user', methods = ['POST'])(user_post)
	api.route('/task/<task_type>', methods = ['GET'])(task_get)
	api.route('/events/<int:clip_id>', methods = ['POST'])(events_post)

	def before_request():
		if db.is_closed():
			db.connect()

	def after_request(response):
		if not db.is_closed():
			db.close()
		return response

	api.before_request(before_request)
	api.teardown_request(after_request)
	sys.argv = ['app', 'app'] + gunicorn_args
	app = type('', (gunicorn.app.wsgiapp.WSGIApplication, ), dict(load = lambda self, *args: api))()
	app.run()

def setup(db_path):
	db = init_db(db_path)
	db.connect()
	db.create_tables([User, Station, Clip, Event])
	print('Database created:', db_path)

def config(environment, root, hostname, website_http_location, resolvers, debug_user_registration_email_file_path, email_authorization_bearer_token):
	conf = jinja2.Template(open(nginx_conf + '.j2').read()).render(environment = environment, root = os.path.abspath(root), hostname = hostname, resolvers = resolvers)
	open(nginx_conf, 'w').write(conf)

	conf = json.loads(open(user_registration_conf + '.j2', 'r').read())
	conf['website_http_location'] = website_http_location
	if email_authorization_bearer_token is not None:
		conf['http']['headers']['Authorization'] = 'Bearer ' + email_authorization_bearer_token
	else:
		conf['debug_user_registration_email_file_path'] = debug_user_registration_email_file_path
	open(user_registration_conf, 'w').write(json.dumps(conf, ensure_ascii = False, indent = 2))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	cmd = subparsers.add_parser('setup')
	cmd.add_argument('--db_path', default = 'app.db')
	cmd.set_defaults(func = setup)

	cmd = subparsers.add_parser('serve')
	cmd.add_argument('--db_path', default = 'app.db')
	cmd.add_argument('--log_sql', action = 'store_true')
	cmd.add_argument('--gunicorn_args', nargs = argparse.REMAINDER, default = [])
	cmd.set_defaults(func = serve)

	cmd = subparsers.add_parser('import')
	cmd.add_argument('--db_path', default = 'app.db')
	cmd.add_argument('--clips_path')
	cmd.add_argument('--stations_path')
	cmd.add_argument('--turnout', action = 'store_true')
	cmd.add_argument('--gold', type = int, default = 0)
	cmd.set_defaults(func = import_)

	cmd = subparsers.add_parser('export')
	cmd.add_argument('--db_path', default = 'app.db')
	cmd.add_argument('--stations_path')
	cmd.set_defaults(func = export_)

	cmd = subparsers.add_parser('config')
	cmd.add_argument('--environment', default = 'development')
	cmd.add_argument('--hostname', default = 'localhost')
	cmd.add_argument('--root', default = '.')
	cmd.add_argument('--resolvers')
	cmd.add_argument('--website_http_location', default = 'http://localhost:8080')
	cmd.add_argument('--debug_user_registration_email_file_path', default = 'debug_user_registration_email.txt')
	cmd.add_argument('--email_authorization_bearer_token')
	cmd.set_defaults(func = config)

	args = vars(parser.parse_args())
	func = args.pop('func')
	func(**args)
