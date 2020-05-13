---
layout: archive 
permalink: /my-projects/
title: "My Projects"
author_profile: true
header:
	image: "/images/6.jpg"

---
{% include base_path %}
{% include group-by-array collection=site.post field="tags" %}

{% for tag in group_name %}
	{% assign posts = group_items[forloop.index0] %}
	<h2 id="{{ tag |slugify }}" class="archive_subtitle">{{ tag }}</h2>
	{% for post in posts %}
		{% include archive-single.html %}
	{% endfor %}
{% endfor %} 