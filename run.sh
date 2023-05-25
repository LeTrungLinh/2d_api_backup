#!/bin/bash
host="0.0.0.0"
port="8001"

uvicorn main:app --host $host --port $port --reload
