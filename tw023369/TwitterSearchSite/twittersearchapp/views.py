from django.shortcuts import render
from django.template import RequestContext, loader
from django.http import HttpResponseRedirect, HttpResponse
import datetime
from twittersearchapp.TwitterAPI import TwitterAPI
from twittersearch.settings import TWITTER_OAUTH_KEY, TWITTER_OAUTH_SECRET
from twittersearchapp.models import SearchPhrase

twitter_api = TwitterAPI()

def display_search(request):
	twitter_api.cache_access_token(TWITTER_OAUTH_KEY, TWITTER_OAUTH_SECRET)

	return render(request, 'search/displaySearch.html', {})

def search(request):
	search_phrase = request.GET.get('search_phrase')

	num_times_searched, last_search_datetime = update_search_model(search_phrase)

	result_list = twitter_api.search(search_phrase);
	context = {'search_phrase': search_phrase,
	           'results': result_list,
	           'num_times_searched': num_times_searched,
	           'last_search_datetime': last_search_datetime}
	return render(request, 'search/searchResults.html', context)

def update_search_model(search_phrase):
    """Update the number of times the phrase has been used to search and the last time it was used
        
    arguments
    search_phrase -- the phrase used for the current search
    """
    model, created = SearchPhrase.objects.get_or_create(text=search_phrase, defaults={'num_times_searched':1})
    last_last_search_datetime = datetime.datetime.now()
    if not created:
    	last_last_search_datetime = model.last_search_datetime
        model.num_times_searched += 1
        model.last_search_datetime = datetime.datetime.now()
        model.save()

    return (model.num_times_searched, last_last_search_datetime)
