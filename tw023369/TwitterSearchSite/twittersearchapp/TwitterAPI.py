import sys
import requests
import base64
from django.utils import timezone
from django.http import HttpRequest
from twittersearchapp.models import SearchPhrase

class TwitterAPI(object):

    """Get the oauth access token

        key - Twitter API oauth key
        secret - Twitter API oauth secret
    """
    def cache_access_token(self, key, secret):
        headers = {'Content-Type' : "application/x-www-form-urlencoded;charset=UTF-8",
                   'Authorization' : "Basic " + base64.urlsafe_b64encode(key + ":" + secret)}
        payload = {'grant_type':"client_credentials"}
        
        response = requests.post("https://api.twitter.com/oauth2/token", params = payload, headers = headers).json()
        
        if 'errors' in response :
            self.raise_twitter_exception(response['errors'])
        if response['token_type'] != "bearer" :
            raise TwitterAPI.TwitterAPIException("Oauth Endpoint returned with no bearer access token")
        self.accessToken = response['access_token']

    """Search tweets for provided phrase
    
        searchPhrase - search phrase
    """
    def search(self, searchPhrase):
        headers = {'Authorization' : "Bearer " + self.accessToken}
        payload = {'q':searchPhrase}
        
        response = requests.get("https://api.twitter.com/1.1/search/tweets.json", params = payload, headers = headers).json()
        
        if 'errors' in response:
            self.raise_twitter_exception(response['errors'])
        return response['statuses']
    
    """Raise an exception with a list of errors

        errors - a list of errors
    """
    def raise_twitter_exception(self, *errors):
        errorString = ""
        for error in errors[0]:
            if 'message' in error:
                errorString = errorString + error['message'] + ", "
        raise TwitterAPI.TwitterAPIException(errorString[:-2])

    class TwitterAPIException(Exception):
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return repr(self.value)
