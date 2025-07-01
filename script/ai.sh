#!/usr/bin/env bash

git pull

python script/ai.py

git add _posts

git commit -am 'update posts'

git push