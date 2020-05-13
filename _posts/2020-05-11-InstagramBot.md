---
title: "Python Project : InstagramBot"
date: 2020-05-11
tags: [Python,Bot]
excerpt: "A Bot that can follow an account and like/comment on posts"
---

## Instagram Bot 

The bot is coded in python and uses selenium (plus its drivers) . I used the firefox as a host for the bot.

**Bot Functionality**

* Opens FireFox browser
* Opens [Instagram.com](https://www.instagram.com/)
* Logs in (provided you've given the username and password)
* Searches for an account (provided you've given the account name)
* Automated Following/Liking/Posting
* Closing the browser

**Requirments :**
* [Python](https://www.python.org/)
* [Python Extension - Selenium](https://pypi.org/project/selenium/)
* [FireFox geckodriver](https://github.com/mozilla/geckodriver/releases)
* An instagram account (So the bot can use the account)
* [FireFox](https://www.mozilla.org/en-US/firefox/new/)

**Download/Clone full code** [here](https://github.com/nvikramraj/Instagram_Bot) 

```python

	print("Code explanation to be added soon")

```

**Warnings** If you get the error element not found in xpath (...) , It is because in the future they may update website changing the xpath or its id . A simple fix is to right click on the element in the page and click on inspect . Then right click on the highlighted part -> copy -> xpath , replace the new xpath by overwritting the old one . 

**References** 
* [Youtube](https://www.youtube.com/watch?v=d2GBO_QjRlo)
* [Geeksforgeeks](https://www.geeksforgeeks.org/like-instagram-pictures-using-selenium-python/)