import jwt
import configs as app_settings
from datetime import timedelta

class MyJwt:
    jwtSecretKey = "UOJNHKzxcxz672bnvghjquvmxjvkjlk"
    jwtAlgorithm = "HS512"
    

    def __init__(self):
        self.jwtSecretKey = app_settings.JWT_SECRET_KEY
    def createJwtToken(self, payloadData):
        # print("jwt secret key : " + self.jwtSecretKey)
        # print("jwt algorithm : " + self.jwtAlgorithm)
        jwtToken = jwt.encode(payloadData, self.jwtSecretKey, algorithm=self.jwtAlgorithm)
        return jwtToken

    def createJtiToken(self, jti, options=None):
        if options is None:
            options = {}
        payloadData = {
            "jti": jti
        }

        if options.get("issuedAt", None) is not None:
            payloadData["iat"] = options.get("issuedAt")

        if options.get("expiration", None) is not None:
            payloadData["exp"] = options.get("expiration")

        jtiToken = jwt.encode(payloadData, self.jwtSecretKey, algorithm=self.jwtAlgorithm)
        return jtiToken

    def decodeJwtToken(self, jwtToken, verifySign=False):
        # print("jwt secret key luc decode : " + self.jwtSecretKey)
        # print("jwt algorithm luc decode : " + self.jwtAlgorithm)
        try:
            decodedData = jwt.decode(jwtToken, self.jwtSecretKey, algorithms=self.jwtAlgorithm, options={"verify_signature": verifySign})
            # print(decodedData)
            return decodedData
        except Exception as ex:
            print("exception in decode jwt token : " + jwtToken)
        return None
