####################Note####################
# ssh-keygen -t rsa -b 4096 -m PEM -f jwtRS256.key
# # Don't add passphrase
# openssl rsa -in jwtRS256.key -pubout -outform PEM -out jwtRS256.key.pub
#############################################
from pathlib import Path
import os 

base_dir = Path(os.getcwd())
img_dir = "images"
JWT_SECRET_KEY = ""
JWT_ALGORITHM = "RS256"
