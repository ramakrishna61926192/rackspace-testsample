from django.conf.urls import patterns, url, include

urlpatterns = patterns('',
	url(r'^twittersearch/', include('twittersearchapp.urls', namespace="twittersearchapp")),
)
