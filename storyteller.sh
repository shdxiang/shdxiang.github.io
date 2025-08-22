#!/usr/bin/env bash

git pull

python3 storyteller.py

git add _posts

git commit -am 'update posts'

git push
