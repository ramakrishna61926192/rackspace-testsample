from django.conf.urls import patterns, url

from twittersearchapp import views

urlpatterns = patterns('',
	url(r'^$', views.display_search, name='displaySearch'),
	url(r'^search/$', views.search, name='search'),
)
