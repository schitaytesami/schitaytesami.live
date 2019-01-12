## Development setup:

```shell
python3 app.py config --config_path nginx.conf.j2 --variables nginx_development.json
./scripts/up start
./scripts/up stop
```

## Production setup:

```shell
python3 app.py config --config nginx.conf.j2 --variables nginx_production.json
sudo systemctl enable <full path to>/schitaytesami.website/systemd/nginx.service
sudo systemctl enable <full path to>/schitaytesami.website/systemd/gunicorn.service
sudo systemctl enable <full path to>/schitaytesami.website/systemd/gunicorn.socket
sudo systemctl start nginx.service
sudo systemctl start gunicorn.service

# restart services
sudo systemctl restart nginx.service
sudo systemctl restart gunicorn.service

# validate config
sudo /usr/sbin/nginx -t -p . -c nginx.conf
```

## Init sequence
```shell
# init app.db
python3 app.py setup
# import clips
python3 app.py import --clips_path https://proverim.webcam/speedup/clips.json --gold 5
# import official turnout
python3 app.py import --stations_path https://github.com/schitaytesami/data/releases/download/20180318/stations.json --turnout
```
