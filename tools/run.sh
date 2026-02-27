#!/usr/bin/env bash
#
# Run Astro dev server

prod=false
command="npm run dev --"
host="127.0.0.1"
port="4321"

help() {
  echo "Usage:"
  echo
  echo "   bash /path/to/run [options]"
  echo
  echo "Options:"
  echo "     -H, --host [HOST]    Host to bind to."
  echo "     -P, --port [PORT]    Port to bind to."
  echo "     -p, --production     Run Astro in production mode preview."
  echo "     -h, --help           Print this help information."
}

while (($#)); do
  opt="$1"
  case $opt in
  -H | --host)
    host="$2"
    shift 2
    ;;
  -P | --port)
    port="$2"
    shift 2
    ;;
  -p | --production)
    prod=true
    shift
    ;;
  -h | --help)
    help
    exit 0
    ;;
  *)
    echo -e "> Unknown option: '$opt'\n"
    help
    exit 1
    ;;
  esac
done

command="$command --host $host --port $port"

if $prod; then
  command="npm run build && npm run preview -- --host $host --port $port"
fi

echo -e "\n> $command\n"
eval "$command"
