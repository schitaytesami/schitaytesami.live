## Principles of current impl
1. Simplicity; ease of deployment and operation
2. Static content over dynamic content; very few requests to backend; minimalistic http backend
3. Immutable database (served well so far)
4. Currently no admin UI; admin works with DB via CLI commands implemented in backend

## Prerequisites
- Linux or Windows/WSL
- nginx
- python3, pip3

Pip dependencies: `pip3 install -r requirements.txt`

## Development setup:
```shell
# generate config file with default development values, user registration emails will be saved in ./debug_user_registration_email.txt
python3 app.py config

# start and stop development web servers
bash ./scripts/up start
bash ./scripts/up stop
```

## Production setup:
```shell
# generate config files
python3 app.py config --environment production --hostname schitaytesami.live --website_http_location 'https://schitaytesami.live' --root /var/www/sami/schitaytesami.live --resolvers "213.133.99.99 213.133.98.98 8.8.8.8" --email_authorization_bearer_token USE_ACTUAL_SENDGRID_BEARER_TOKEN_HERE

sudo systemctl enable $PWD/systemd/schitaytesami.nginx.service
sudo systemctl enable $PWD/systemd/schitaytesami.gunicorn.service
sudo systemctl enable $PWD/systemd/schitaytesami.gunicorn.socket
sudo systemctl start schitaytesami.nginx.service
sudo systemctl start schitaytesami.gunicorn.service

# restart services
sudo systemctl restart schitaytesami.nginx.service
sudo systemctl restart schitaytesami.gunicorn.service

# validate config
sudo /usr/sbin/nginx -t -p . -c nginx.conf
```

## Database init
```shell
# init app.db
python3 app.py setup

# add user
python3 app.py add_user

# add admin
python3 app.py add_user --admin

# import clips
python3 app.py import --clips_path https://github.com/schitaytesami/schitaytesami.live/releases/download/data/clips.json --gold 5

# import official turnout
python3 app.py import --stations_path https://github.com/schitaytesami/data/releases/download/20180318/stations.json --turnout
```
