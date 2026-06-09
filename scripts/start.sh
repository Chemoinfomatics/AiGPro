#! /usr/bin/env sh

# exec uvicorn $APP_MODULE --port $PORT --host $HOST --app-dir dashboard/api/ --root-path /api/ --forwarded-allow-ips "*" --reload --proxy-headers &
# exec uvicorn $APP_MODULE --port $PORT --host $HOST --app-dir dashboard/api/ --root-path /api/  --reload --proxy-headers &
# echo something

exec uvicorn $APP_MODULE --port $PORT --host $HOST --proxy-headers --reload
