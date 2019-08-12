import argparse
import base64
import bcrypt
import binascii
import flask
import functools
import gunicorn.app.wsgiapp
import itertools
import jinja2
import json
import logging
import os
import peewee as pw
import random
import requests
import string
import sys
import time
import urllib.request

DEFAULT_DB_PATH = 'app.db'
DEFAULT_IMPORT_BATCH_SIZE = 32
DEFAULT_WEBSITE_URL = 'http://localhost:8080'
DEFAULT_DEBUG_USER_EMAIL_PATH = 'debug_user_email.txt'
NGINX_CONF_PATH = 'nginx.conf'
USER_REGISTRATION_CONF = 'user_registration.json'

incomplete_task_threshold = dict(vote=4)


# ================================================== #


class User(pw.Model):
    id = pw.PrimaryKeyField()
    display = pw.TextField()
    email = pw.TextField(default='')
    email_normalized = pw.TextField(default='')
    password = pw.TextField(default=lambda: binascii.hexlify(os.urandom(8)).decode('ascii'), null=True)
    hashed = pw.TextField(null=True)
    role = pw.TextField(default='task/vote')

    @property
    def is_admin(self):
        return 'admin' in self.role

    def hash_password(self):
        self.hashed = bcrypt.hashpw(self.password.encode('utf-8'), bcrypt.gensalt())
        self.password = None

    def verify_password(self, user):
        return bcrypt.checkpw(self.password.encode('utf-8'), user.hashed.encode('utf-8'))

    def generate_token(self):
        token_data = dict(id=self.id, display=self.display, role=self.role, password=self.password)
        return base64.b64encode(json.dumps(token_data).encode('ascii')).decode('ascii').rstrip('=')

    def parse_token(self, user_token):
        parsed = json.loads(base64.b64decode(user_token + '===').decode('ascii'))
        self.id, self.display, self.role, self.password = \
            map(parsed.get, ['id', 'display', 'role', 'password'])

    def send_registration_email(self, user_token, email_subject, email_body, sender_email, sender_name, website_url,
                                debug_user_email_path, endpoint, headers, data):
        email_body = string.Template(email_body).substitute(user_token=user_token, website_url=website_url)
        payload = string.Template(data).substitute(registration_email_subject=email_subject,
                                                   registration_email_body=email_body,
                                                   sender_email=sender_email,
                                                   sender_name=sender_name,
                                                   recipient=self.email)
        if debug_user_email_path is not None:
            open(debug_user_email_path, 'w').write(payload)
        requests.post(endpoint, headers=headers, data=payload.encode('utf-8'))

    @staticmethod
    def normalize_email(email):
        splits = email.split('@')
        return splits[0].split('+')[0].replace('.', '') + '@' + (splits[1] if len(splits) >= 2 else '')

    @staticmethod
    def generate_display(nicknames, lo=100, hi=1000):
        return random.choice(nicknames) + '_' + str(random.randint(lo, hi))


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
    station = pw.ForeignKeyField(Station, backref='clips')
    video = pw.TextField(null=True)
    thumbnail = pw.TextField(null=True)
    clip_interval_start = pw.IntegerField(null=True)
    clip_interval_end = pw.IntegerField(null=True)
    meta = pw.TextField(default='')
    task = pw.TextField(default='', index=True)
    camera_id = pw.TextField(default='')
    csrf = pw.IntegerField(default=lambda: random.randint(0, 1e9))
    gold = pw.BooleanField(default=False)


class Event(pw.Model):
    creator = pw.ForeignKeyField(User, null=True)
    timestamp = pw.IntegerField(default=lambda: int(time.time()))
    clip = pw.ForeignKeyField(Clip, null=True, backref='events')
    station = pw.ForeignKeyField(Station, null=True, backref='events')
    value = pw.TextField()
    offset = pw.DoubleField(default=0.0)
    type = pw.TextField(index=True)


class StationAccess(pw.Model):
    station = pw.ForeignKeyField(Station)
    user = pw.ForeignKeyField(User)
    timestamp = pw.IntegerField(default=lambda: int(time.time()))
    granted = pw.BooleanField(default=False)


# ================================================== #


def build_response_(data):
    return flask.Response(response=json.dumps(data, ensure_ascii=False),
                          status=200,
                          mimetype='application/json')


def user_id_from_event_(event):
    return event.creator_id


def clip_id_from_event_(event):
    return event.clip_id


def station_id_from_event_(event):
    if event.station is not None:
        return event.station_id
    elif event.clip is not None:
        return event.clip.station_id
    return None


def user_must_be_active(admin=False, fail=True):
    def wrap(method):
        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            try:
                user_stub = User()
                user_stub.parse_token(flask.request.cookies['user_token'])
                user = User.get_or_none(User.id == user_stub.id)
                if user is None or not user_stub.verify_password(user) or (admin and not user.is_admin):
                    raise Exception()
            except:
                user = None
                if fail:
                    flask.abort(401)
            return method(user=user, *args, **kwargs)

        return wrapped

    return wrap


def user_post():
    settings = json.load(open(USER_REGISTRATION_CONF))
    email = flask.request.data.decode('utf-8')
    email_normalized = User.normalize_email(email)
    if User.get_or_none(User.email_normalized == email_normalized) is None:
        user = User.create(email=email,
                           email_normalized=email_normalized,
                           display=User.generate_display(settings['nicknames']))
        user_token = user.generate_token()
        user.hash_password()
        user.save()
        user.send_registration_email(user_token=user_token,
                                     email_subject=settings['registration_email_subject'],
                                     email_body=settings['registration_email_body'],
                                     sender_email=settings['sender_email'],
                                     sender_name=settings['sender_name'],
                                     website_url=settings['website_url'],
                                     debug_user_email_path=settings['debug_user_email_path'],
                                     **settings['http'])
    return flask.jsonify(success=True)


@user_must_be_active()
def events_post(clip_id, user):
    events = flask.request.get_json()
    if len(events) > 0 and Clip.get(Clip.id == clip_id).csrf == int(flask.request.args.get('csrf', 0)):
        event_to_insert = list()
        for event in events:
            if event.get('clip') == clip_id:
                event_dict = dict(creator=user,
                                  clip=event.get('clip'),
                                  type=event['type'],
                                  value=event.get('value', ''),
                                  offset=event['offset'])
                event_to_insert.append(event_dict)
        Event.insert_many(event_to_insert).execute()
    return flask.jsonify(success=True)


@user_must_be_active()
def user_access_station_post(user_id, station_id, user):
    if user.is_admin or user_id == user.id:
        StationAccess.create(station_id=station_id, user_id=user_id, granted=user.is_admin).save()
        return flask.jsonify(success=True)
    else:
        return flask.jsonify(success=False)


def estimate_clip_turnout_(clip):
    vote_events = sorted(filter(lambda event: event.type == 'vote', clip.events), key=user_id_from_event_)
    votes = [len(list(g)) for k, g in itertools.groupby(vote_events, key=user_id_from_event_)]
    final_turnout = ([json.loads(event.value) for event in clip.events if event.type == 'turnout_estimate'] + [None])[0]
    average_turnout = int(sum(votes) / float(len(votes))) if len(votes) > 0 else None
    completed = len(votes) >= incomplete_task_threshold['vote']
    clip_turnout = dict(final=final_turnout, average=average_turnout, completed=completed)
    return clip_turnout


def interval_turnout_(station=None, clip_turnout=None, official=None, hours_begin=None, hours_end=None,
                      timestamp_begin=None, timestamp_end=None, hours_baseline=8, normalize=True):
    # Get interval's start and end timestamps.
    if hours_begin is not None and hours_end is not None:
        baseline_timestamp = station.station_interval_start - hours_baseline * 60 * 60
        interval_start = baseline_timestamp + hours_begin * 60 * 60
        interval_end = baseline_timestamp + hours_end * 60 * 60
    else:
        interval_start = timestamp_begin
        interval_end = timestamp_end
    # Get turnout counts for the computed interval.
    clip_turnout_in = list()
    for clip in station.clips:
        if (interval_start <= clip.clip_interval_start and clip.clip_interval_end <= interval_end) or \
                (clip.clip_interval_start <= interval_end <= clip.clip_interval_end):
            clip_turnout_in.append(clip_turnout[clip.id])
    # Get the final count and normalize by the official count, if needed.
    if all(turnout['final'] is not None for turnout in clip_turnout_in):
        final_count = sum(turnout['final']['count'] for turnout in clip_turnout_in)
    else:
        final_count = None
    if not normalize:
        return final_count
    official_count = official.get('voters_registered')
    if final_count is not None and official_count is not None:
        return final_count / official_count
    return None


def estimate_station_turnout_(station, clip_turnout):
    # Calculate official turnout of the given station.
    official_turnout = [json.loads(event.value) for event in station.events if event.type == 'turnout_official']
    official_turnout = (official_turnout + [{}])[0]
    # Calculate estimate turnout of the given station.
    estimate_turnout = {'10h': interval_turnout_(station=station, clip_turnout=clip_turnout, official=official_turnout,
                                                 hours_begin=8, hours_end=10),
                        '12h': interval_turnout_(station=station, clip_turnout=clip_turnout, official=official_turnout,
                                                 hours_begin=10, hours_end=12),
                        '15h': interval_turnout_(station=station, clip_turnout=clip_turnout, official=official_turnout,
                                                 hours_begin=12, hours_end=15),
                        '18h': interval_turnout_(station=station, clip_turnout=clip_turnout, official=official_turnout,
                                                 hours_begin=15, hours_end=18),
                        '20h': interval_turnout_(station=station, clip_turnout=clip_turnout, official=official_turnout,
                                                 hours_begin=18, hours_end=20),
                        'final': interval_turnout_(station=station,
                                                   clip_turnout=clip_turnout,
                                                   official=official_turnout,
                                                   timestamp_begin=station.station_interval_start,
                                                   timestamp_end=station.station_interval_end,
                                                   normalize=False)}
    # Calculate progress of the given station.
    clip_turnouts = [clip_turnout[clip.id] for clip in station.clips]
    progress = sum(1 if turnout['completed'] else 0 for turnout in clip_turnouts) / float(len(clip_turnouts))
    comment = ' || '.join(turnout['final']['comment']
                          for turnout in clip_turnouts
                          if (turnout['final'] or {}).get('comment') not in ['', None])
    return dict(estimate=estimate_turnout, official=official_turnout, comment=comment, progress=progress)


# TODO: Refactor and simplify this.
def build_task_selector_options_(stations):
    groupby = lambda stations, key: [(k, list(g)) for k, g in itertools.groupby(sorted(stations, key=key), key=key)]
    by_station = lambda stations: [('УИК #{}'.format(station.station_number), station.station_number)
                                   for station in sorted(stations, key=lambda station: station.station_number)]
    by_region = lambda stations: [('Регион: Автоматический выбор', -1, [('УИК: Автоматический выбор', -1)])] + \
                                 [(g[0].station_address.split(',')[0], k,
                                   [('УИК: Автоматический выбор', -1)] + by_station(g))
                                  for k, g in groupby(stations, key=lambda station: station.region_number)]
    by_election = lambda stations: [('Выборы {}'.format(k), k, by_region(g))
                                    for k, g in groupby(stations, key=lambda station: station.election_number)]
    return by_election(stations)


def stats_get():
    stations = list(Station.select()
                    .order_by(Station.election_number, Station.region_number, Station.station_number)
                    .prefetch(Event, Clip))
    clips = list(Clip.select().where(Clip.task == 'vote').prefetch(Event, Station))

    # Collect turnout counts.
    clip_turnout = {clip.id: estimate_clip_turnout_(clip) for clip in clips}
    station_turnout = {station.id: estimate_station_turnout_(station, clip_turnout) for station in stations}

    # Collect stations.
    stations_list = list()
    for station in stations:
        station_dict = dict(id=station.id,
                            station_id=station.station_id,
                            election_number=station.election_number,
                            region_number=station.region_number,
                            station_number=station.station_number,
                            station_address=station.station_address,
                            timezone_offset=station.timezone_offset,
                            station_interval_start=station.station_interval_start,
                            station_interval_end=station.station_interval_end,
                            turnout=station_turnout[station.id],
                            clips=[c.id for c in station.clips])
        stations_list.append(station_dict)

    # Collect clips.
    clips_list = list()
    for clip in clips:
        clip_dict = dict(id=clip.id,
                         station_id=clip.station_id,
                         video=clip.video,
                         thumbnail=clip.thumbnail,
                         clip_interval_start=clip.clip_interval_start,
                         clip_interval_end=clip.clip_interval_end,
                         turnout=clip_turnout[clip.id],
                         camera_id=clip.camera_id,
                         task=clip.task)
        clips_list.append(clip_dict)

    # Collect bookmarks.
    bookmarks_list = list()
    for event in Event.select().where(Event.type == 'bookmark').prefetch(Clip):
        bookmark_dict = dict(id=event.id,
                             timestamp=event.timestamp,
                             value=event.value,
                             station_id=station_id_from_event_(event))
        bookmarks_list.append(bookmark_dict)

    # Collect notes.
    notes_list = list()
    for event in Event.select().where(Event.type == 'note').prefetch(Clip):
        notes_dict = dict(id=event.id,
                          timestamp=event.timestamp,
                          value=event.value,
                          station_id=station_id_from_event_(event))
        notes_list.append(notes_dict)

    # Collect station accesses.
    station_access_list = list()
    station_accesses = StationAccess.raw(
        'SELECT a.user_id, a.station_id, MAX(a.granted) as granted, MAX(a.timestamp) as timestamp '
        'FROM StationAccess a '
        'GROUP BY a.user_id, a.station_id '
        'ORDER BY granted ASC, timestamp DESC'
    )
    for station_access in station_accesses:
        station_access_dict = dict(user_id=station_access.user_id,
                                   station_id=station_access.station_id,
                                   timestamp=station_access.timestamp,
                                   granted=1 if station_access.granted else 0)
        station_access_list.append(station_access_dict)

    # Collect users.
    users_list = list()
    users = User.raw(
        'SELECT '
        '   u.id, '
        '   u.display, '
        '   IFNULL(SUM(e.type == "vote"), 0) as num_votes, '
        '	IFNULL(SUM(e.type == "note"), 0) as num_notes, '
        '	IFNULL(COUNT(DISTINCT c.station_id), 0) as num_stations, '
        '	SUM(IFNULL(c.clip_interval_end - c.clip_interval_start, 0)) as num_seconds, '
        '	IFNULL(COUNT(DISTINCT e.clip_id), 0) as num_clips, '
        '	IFNULL(GROUP_CONCAT(DISTINCT e.clip_id), "") as clips, '
        '	IFNULL(GROUP_CONCAT(CASE e.type WHEN "note" THEN e.id ELSE NULL END, ","), "") as notes, '
        '	IFNULL(GROUP_CONCAT(CASE e.type WHEN "bookmark" THEN e.id ELSE NULL END, ","), "") as bookmarks '
        'FROM User u '
        'LEFT OUTER JOIN Event e ON e.creator_id = u.id '
        'LEFT OUTER JOIN Clip c ON e.clip_id = c.id '
        'GROUP BY u.id, u.display '
        'ORDER BY num_votes DESC'
    )
    for user_dict in users.dicts():
        for key in ['notes', 'bookmarks', 'clips']:
            user_dict[key] = list(map(int, user_dict[key].split(','))) if user_dict[key] else []
        users_list.append(user_dict)

    # Collect other stats.
    num_stations_labeled = sum(1 for t in station_turnout.values() if t['estimate'].get('final') is not None)
    num_seconds = Clip._meta.database.execute_sql(
        'SELECT IFNULL(SUM(IFNULL(c.clip_interval_end, 0) - IFNULL(c.clip_interval_start, 0)), 0) '
        'FROM Clip c '
        'WHERE c.task == "vote"'
    ).fetchone()[0]
    num_seconds_labeled = Clip._meta.database.execute_sql(
        'SELECT IFNULL(SUM(IFNULL(c.clip_interval_end, 0) - IFNULL(c.clip_interval_start, 0)), 0) '
        'FROM Clip c '
        'INNER JOIN Event e ON e.clip_id == c.id AND e.type == "vote" '
        'WHERE c.task == "vote"'
    ).fetchone()[0]

    # Merge all data into a single dict.
    stats_dict = dict(
        stations=stations_list,
        clips=clips_list,
        bookmarks=bookmarks_list,
        notes=notes_list,
        station_access=station_access_list,
        users=users_list,
        num_stations_labeled=num_stations_labeled,
        num_seconds=num_seconds,
        num_seconds_labeled=num_seconds_labeled,
        task_selector_options=build_task_selector_options_(stations)
    )
    return build_response_(stats_dict)


def verify_station_access_(station, user):
    if user.is_admin:
        return True
    for access in StationAccess.select().where(StationAccess.station == station.id and StationAccess.user == user.id):
        if access is not None and access.granted is True:
            return True
    return False


@user_must_be_active()
def station_get(station_id, user):
    station = Station.get_or_none(Station.id == station_id)
    if station is None:
        return build_response_(None)
    # Collect all events for the given station and user. Verify access.
    events = list()
    if verify_station_access_(station, user):
        for event in Event.select().where(Event.creator_id == user.id).prefetch(Clip):
            if station_id_from_event_(event) == station.id:
                events.append(dict(id=event.id,
                                   timestamp=event.timestamp,
                                   value=event.value,
                                   station_id=station_id_from_event_(event),
                                   clip_id=clip_id_from_event_(event)))
    # Collect clips for the given station.
    clips = list()
    for clip_id, clip_events_group in itertools.groupby(events, lambda x: x['clip_id']):
        clips.append(dict(id=clip_id, events=list(clip_events_group)))
    # Build response.
    station_dict = dict(id=station.id,
                        station_id=station.station_id,
                        station_number=station.station_number,
                        region_number=station.region_number,
                        election_number=station.election_number,
                        station_address=station.station_address,
                        timezone_offset=station.timezone_offset,
                        station_interval_start=station.station_interval_start,
                        station_interval_end=station.station_interval_end,
                        clips=clips)
    return build_response_(station_dict)


def build_filter_sql_(election_number, region_number, station_number):
    filter_sql, filter_sql_args = '', []
    for k, v in dict(election_number=election_number,
                     region_number=region_number,
                     station_number=station_number).items():
        if v is not None and int(v) >= 0:
            filter_sql += ' AND s.{} == ? '.format(k)
            filter_sql_args += [int(v)]
    return filter_sql, filter_sql_args


@user_must_be_active()
def task_get(task_type, election_number, region_number, station_number, user, limit_num=20):
    # Get all available incomplete tasks (unprocessed clips) for the given station and user.
    filter_sql, filter_sql_args = build_filter_sql_(election_number, region_number, station_number)
    incomplete_tasks = list(Clip.raw(
        'SELECT c.*, '
        'IFNULL(COUNT(DISTINCT e.creator_id), 0) as num_completed, '
        'IFNULL(MAX(e.creator_id == ?), 0) as is_completed '
        'FROM Clip c '
        'JOIN Station s ON c.station_id = s.id '
        'LEFT OUTER JOIN Event e ON e.clip_id == c.id AND e.type == "vote" '
        'WHERE c.task == ? ' + filter_sql + ' ' +
        'GROUP BY c.id, c.station_id, c.video, c.thumbnail, c.clip_interval_start, c.clip_interval_end, '
        's.election_number, s.region_number ,s.station_number '
        'HAVING is_completed == 0 AND num_completed < ? '
        'ORDER BY c.gold DESC, num_completed DESC, s.election_number ASC, s.region_number ASC, s.station_number ASC, '
        'c.clip_interval_start ASC, c.clip_interval_end ASC '
        'LIMIT ?',
        *([user.id, task_type] + filter_sql_args + [incomplete_task_threshold[task_type], limit_num])
    )) or []
    # Choose next task (clip) randomly.
    clip = random.choice(incomplete_tasks[:limit_num])
    station_dict = dict(id=clip.station_id,
                        station_number=clip.station.station_number,
                        station_address=clip.station.station_address,
                        timezone_offset=clip.station.timezone_offset,
                        election_number=clip.station.election_number)
    task_dict = dict(id=clip.id,
                     task=clip.task,
                     thumbnail=clip.thumbnail,
                     video=clip.video,
                     clip_interval_start=clip.clip_interval_start,
                     clip_interval_end=clip.clip_interval_end,
                     csrf=clip.csrf,
                     station=station_dict)
    return build_response_(task_dict)


# ================================================== #


def init_db_(db_path):
    db = pw.SqliteDatabase(db_path, autocommit=False)
    db.bind([User, Station, Clip, Event, StationAccess])
    return db


def setup(db_path):
    db = init_db_(db_path)
    db.connect()
    db.create_tables([User, Station, Clip, Event, StationAccess])
    print('DONE:', db_path)


def config(environment, hostname, root, resolvers, website_url, debug_user_email_path, email_auth_token):
    # Generate NGINX config.
    base_nginx_conf = jinja2.Template(open(NGINX_CONF_PATH + '.j2').read())
    nginx_conf = base_nginx_conf.render(environment=environment,
                                        root=os.path.abspath(root),
                                        hostname=hostname,
                                        resolvers=resolvers)
    open(NGINX_CONF_PATH, 'w').write(nginx_conf)
    # Generate user registration config.
    email_conf = json.loads(open(USER_REGISTRATION_CONF + '.j2', 'r').read())
    email_conf['website_url'] = website_url
    email_conf['debug_user_email_path'] = debug_user_email_path
    if email_auth_token is not None:
        email_conf['http']['headers']['Authorization'] = 'Bearer ' + email_auth_token
    open(USER_REGISTRATION_CONF, 'w').write(json.dumps(email_conf, ensure_ascii=False, indent=2))
    print('DONE')


def serve(db_path, log_sql, gunicorn_args):
    db = init_db_(db_path)
    if log_sql:
        logger = logging.getLogger('peewee')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

    api = flask.Flask(__name__, static_url_path='')
    api.route('/user', methods=['POST'])(user_post)
    api.route('/events/<int:clip_id>', methods=['POST'])(events_post)
    api.route('/user/<int:user_id>/access/station/<int:station_id>', methods=['POST'])(user_access_station_post)
    api.route('/stats', methods=['GET'])(stats_get)
    api.route('/station/<int:station_id>', methods=['GET'])(station_get)
    api.route('/task/<task_type>', methods=['GET'],
              defaults=dict(election_number=-1, region_number=-1, station_number=-1))(task_get)
    api.route('/task/<task_type>/election/<election_number>/region/<region_number>/station/<station_number>',
              methods=['GET'])(task_get)

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
    app = type('', (gunicorn.app.wsgiapp.WSGIApplication,), dict(load=lambda self, *args: api))()
    app.run()
    print('DONE')


def add_user(email, admin, db_path):
    init_db_(db_path)
    settings = json.loads(open(USER_REGISTRATION_CONF, 'r').read())
    user = User.create(email=email,
                       email_normalized=User.normalize_email(email),
                       display=User.generate_display(settings['nicknames']),
                       role='admin' if admin else '')
    user_token = user.generate_token()
    user.hash_password()
    user.save()
    print('DONE: {}/#login/{}'.format(settings['website_url'], user_token))


def import_data(db_path, clips_path, stations_path, turnout, gold, batch_size):
    def load_json(uri):
        contents = open(uri, 'rb') if not uri.startswith('http') else urllib.request.urlopen(uri)
        return json.loads(contents.read().decode('utf-8'))

    with init_db_(db_path).atomic():
        # Import clips.
        if clips_path is not None and turnout is False:
            clips_to_insert = list()
            imported_clips = load_json(clips_path)
            for clip in imported_clips:
                station = Station.get_or_create(station_number=clip['station_number'],
                                                region_number=clip['region_number'],
                                                election_number=clip['election_number'],
                                                defaults=dict(station_address=clip['station_address'],
                                                              timezone_offset=clip['timezone_offset'],
                                                              station_interval_start=clip['station_interval_start'],
                                                              station_interval_end=clip['station_interval_end']))[0]
                clip_dict = dict(clip_interval_start=clip.get('clip_interval_start'),
                                 clip_interval_end=clip.get('clip_interval_end'),
                                 video=clip.get('video'),
                                 thumbnail=clip.get('thumbnail'),
                                 meta=clip.get('meta', ''),
                                 station_id=station.id,
                                 task=clip.get('task', ''),
                                 camera_id=clip.get('camera_id' ''),
                                 gold=clip.get('gold', random.random() < float(gold) / len(imported_clips)))
                clips_to_insert.append(clip_dict)
            for batch in pw.chunked(clips_to_insert, batch_size):
                Clip.insert_many(batch).execute()

        # Import turnout events from the imported clips.
        if clips_path is not None and turnout is True:
            events_to_insert = list()
            imported_clips = {clip['id']: clip for clip in load_json(clips_path)}
            for clip in list(Clip.select().prefetch(Event)):
                if not any(event.type == 'turnout_estimate' for event in clip.events):
                    if clip.id in imported_clips:
                        event_dict = dict(clip=clip,
                                          type='turnout_estimate',
                                          value=json.dumps(imported_clips[clip.id], ensure_ascii=False))
                        events_to_insert.append(event_dict)
            for batch in pw.chunked(events_to_insert, batch_size):
                Event.insert_many(batch).execute()

        # Import turnout events from the imported stations.
        if stations_path is not None and turnout is True:
            events_to_insert = list()
            imported_stations = {station['station_id']: station for station in load_json(stations_path)}
            for station in list(Station.select().prefetch(Event)):
                if not any(event.type == 'turnout_official' for event in station.events):
                    if station.station_id in imported_stations:
                        station_turnout = imported_stations[station.station_id]
                        turnout_hist = {'10h': station_turnout['turnout_10h'],
                                        '12h': station_turnout['turnout_12h'],
                                        '15h': station_turnout['turnout_15h'],
                                        '18h': station_turnout['turnout_18h'],
                                        '20h':
                                            float(station_turnout['ballots_given_at_station_on_election_day']) /
                                            float(station_turnout['voters_registered']),
                                        'final': station_turnout['ballots_given_at_station_on_election_day']}
                        event_dict = dict(station=station,
                                          type='turnout_official',
                                          value=json.dumps(turnout_hist, ensure_ascii=False))
                        events_to_insert.append(event_dict)
            for batch in pw.chunked(events_to_insert, batch_size):
                Event.insert_many(batch).execute()
    print('DONE')


def export_data(db_path, stations_path):
    init_db_(db_path)
    exported_stations = list()
    for station in Station.select().prefetch(Clip, Event, User):
        exported_clips = list()
        for clip in station.clips:
            exported_events = list()
            for event in clip.events:
                event_dict = dict(id=event.id,
                                  creator=event.creator.display,
                                  timestamp=event.timestamp,
                                  clip=event.clip_id,
                                  station=event.station_id,
                                  value=event.value,
                                  offset=event.offset,
                                  type=event.type)
                exported_events.append(event_dict)
            clip_dict = dict(camera_id=clip.camera_id,
                             thumbnail=clip.thumbnail,
                             task=clip.task,
                             completed=estimate_clip_turnout_(clip)['completed'],
                             events=exported_events)
            exported_clips.append(clip_dict)
        station_dict = dict(id=station.id,
                            station_number=station.station_number,
                            region_number=station.region_number,
                            election_number=station.election_number,
                            station_address=station.station_address,
                            timezone_offset=station.timezone_offset,
                            station_interval_start=station.station_interval_start,
                            station_interval_end=station.station_interval_end,
                            clips=exported_clips)
        exported_stations.append(station_dict)
    json.dump(exported_stations, open(stations_path, 'w'), ensure_ascii=False, indent=2, sort_keys=True)
    print('DONE')


# ================================================== #


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    cmd = subparsers.add_parser('setup')
    cmd.add_argument('--db_path', default=DEFAULT_DB_PATH)
    cmd.set_defaults(func=setup)

    cmd = subparsers.add_parser('config')
    cmd.add_argument('--environment', default='development')
    cmd.add_argument('--hostname', default='localhost')
    cmd.add_argument('--root', default='.')
    cmd.add_argument('--resolvers')
    cmd.add_argument('--website_url', default=DEFAULT_WEBSITE_URL)
    cmd.add_argument('--debug_user_email_path', default=DEFAULT_DEBUG_USER_EMAIL_PATH)
    cmd.add_argument('--email_auth_token')
    cmd.set_defaults(func=config)

    cmd = subparsers.add_parser('serve')
    cmd.add_argument('--db_path', default=DEFAULT_DB_PATH)
    cmd.add_argument('--log_sql', action='store_true')
    cmd.add_argument('--gunicorn_args', nargs=argparse.REMAINDER, default=[])
    cmd.set_defaults(func=serve)

    cmd = subparsers.add_parser('add_user')
    cmd.add_argument('--db_path', default=DEFAULT_DB_PATH)
    cmd.add_argument('--email', default='{}@testuser.com'.format(random.randint(10, 100)))
    cmd.add_argument('--admin', action='store_true')
    cmd.set_defaults(func=add_user)

    cmd = subparsers.add_parser('import')
    cmd.add_argument('--db_path', default=DEFAULT_DB_PATH)
    cmd.add_argument('--clips_path')
    cmd.add_argument('--stations_path')
    cmd.add_argument('--turnout', action='store_true')
    cmd.add_argument('--gold', type=int, default=0)
    cmd.add_argument('--batch_size', type=int, default=DEFAULT_IMPORT_BATCH_SIZE)
    cmd.set_defaults(func=import_data)

    cmd = subparsers.add_parser('export')
    cmd.add_argument('--db_path', default=DEFAULT_DB_PATH)
    cmd.add_argument('--stations_path')
    cmd.set_defaults(func=export_data)

    cmd_args = vars(parser.parse_args())
    func = cmd_args.pop('func')
    func(**cmd_args)
