from django.db import models

class SearchPhrase(models.Model):
	text = models.CharField(max_length=100)
	num_times_searched = models.IntegerField(default=0)
	last_search_datetime = models.DateTimeField('last searched', auto_now=True)
