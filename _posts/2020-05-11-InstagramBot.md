---
title: "Python Project : InstagramBot"
date: 2020-05-11
tags: [Python,Bot]
excerpt: "A Bot that can follow an account and like/comment on posts"
---

## Instagram Bot 

The bot is coded in python and uses selenium (plus its drivers) . I used the firefox as a host for the bot.

**Bot Functionality**

* Opens FireFox browser.
* Opens [Instagram.com](https://www.instagram.com/)
* Logs in (provided you've given the username and password).
* Searches for an account (provided you've given the account name).
* Automated Following/Liking/Posting.
* Closing the browser.

**Requirments :**
* [Python](https://www.python.org/)
* [Python Extension - Selenium](https://pypi.org/project/selenium/)
* [FireFox geckodriver](https://github.com/mozilla/geckodriver/releases)
* An instagram account (So the bot can use the account)
* [FireFox](https://www.mozilla.org/en-US/firefox/new/)

**Download/Clone full code** [here](https://github.com/nvikramraj/Instagram_Bot) 

**Stuff to import before coding**

```python

from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
from time import sleep #inbuilt function 

```

The xpaths used in this code will work for firefox . The delay/sleep time can be customised .

# Launching FireFox :

To launch the browser , you will need to give the path to geckodriver (dl link above) .
eg :  X:\my_bot\geckodriver.exe

```python 

exe_path = input("Enter the path of your driver : ")
global browser
browser = webdriver.Firefox(executable_path = exe_path) #Gets the geckodriver

```

# Opening Instagram and logging in :

To log in on Instagram , you will need to find the xpath 3 elements :

1. Username
2. Password
3. Submit button 

The bot will automatically fill the username , password and press the submit button. 
When you log in , the page will ask you to turn on notification everytime . To turn it off find the xpath of the not now button and click it (code is included below).

```python

class insta_bot():

	def __init__(self,usrname,pswrd):
		#going to the website
		browser.get("http://instagram.com")
		sleep(2)
		#finding the username and password input area and filling them
		logindets = browser.find_element_by_xpath("//input[@name=\"username\"]")\
			.send_keys(usrname)
		logindets = browser.find_element_by_xpath("//input[@name=\"password\"]")\
			.send_keys(pswrd)
		#the button to login is "submit" type , so using that to find it and clicking it
		logindets = browser.find_element_by_xpath('//button[@type="submit"]')\
			.click()
		sleep(5)
		#this is used to click on Not now button in the notifications , can be removed if its not shown
		browser.find_element_by_xpath("//button[contains(text(), 'Not Now')]")\
            .click()
		sleep(3)

username = input("Enter the username of your account : ")
password = input("Enter the password of your account : ")
my_bot = insta_bot(username,password)

```
I've used a constructor in class to launch instagram and log in . But it is not necessary , it works perfectly fine outside a class also.

# To find an account :

To search for the account , you need to find the search bar's xpath.
The bot will fill the name in search bar and press Return twice , The page will be directed to that account.

```python

	def finduser(self,name):
		#finding the search bar and entering name 
		browser.find_element_by_xpath("/html/body/div[1]/section/nav/div[2]/div/div/div[2]/input")\
			.send_keys(name + Keys.RETURN)
		sleep(2)
		#to click on the searched name
		browser.find_element_by_xpath("//a[contains(@href,'/{}')]".format(name))\
			.send_keys(Keys.RETURN)
		sleep(5)

like_acc = input("Enter the Account name : ")
my_bot.finduser(like_acc)

```

# Following an account :

Find the xpath of follow button and click on it.

```python

def follow():
	sleep(1)
	browser.find_element_by_xpath("//button[contains(text(), 'Follow')]").click()

follow()

```

# Liking / Commenting on a Post :

After going to an account , to like/comment on a post . Click on the first post (by finding the xpath).

```python

def first_post(): 
	# finds the first post 
	pic = browser.find_element_by_class_name("_9AhH0") 
	pic.click() # clicks on the first post 

first_post() 

```

To like the post , find the xpath of the like button and click it.

```python

def like_post(): 
	sleep(2) 
	like = browser.find_element_by_xpath('/html/body/div[4]/div[2]/div/article/div[2]/section[1]/span[1]/button') 
	# finding the like button
	sleep(2) 
	like.click() # clicking the like button

like_post()

```

To comment on the post , find the xpath of the comment box and click on it . Fill the message and click on post.

```python

def comment_post(message):
	sleep(2)
	#finds the comment's area and clicks on it
	comment = browser.find_element_by_xpath('/html/body/div[4]/div[2]/div/article/div[2]/section[3]/div/form/textarea')
	comment.click()
	sleep(2)
	#Types the comment
	browser.find_element_by_xpath('/html/body/div[4]/div[2]/div/article/div[2]/section[3]/div/form/textarea').send_keys(message)
	sleep(1)
	#Posts the comment
	browser.find_element_by_xpath("//button[contains(text(), 'Post')]").click()

msg = input("Enter the comment you want to spam : ")
comment_post(msg)

```  
# To like/comment on all posts :

To do this we need a way to go to the next post , Again all we have to do is find the xpath of next button and click it.

```python

def next_post(): 
	sleep(2) 
	# finds the button which gives the next post 
	nex = browser.find_element_by_xpath('/html/body/div[4]/div[1]/div/div/a[2]')
	sleep(1) 
	return nex 

```

Call the like/comment and next function using a loop to spam likes/comments

```python

def like_till_the_end(): 
	next_el = browser.find_element_by_xpath('/html/body/div[4]/div[1]/div/div/a')
	#The xpath of next in first post is different from others
	while(True): 
		# if next button is there then 
		if next_el != False: 
			# click the next button 
			next_el.click() 
			sleep(2) 
			# like the post 
			like_post()	 
			sleep(2)			 
		else: 
			print("The End")  #it will show an error at the end , recommended to fix it 
			break
		next_el = next_post()

def comment_till_the_end(): 
	next_el = browser.find_element_by_xpath('/html/body/div[4]/div[1]/div/div/a')
	#The xpath of next in first post is different from others
	while(True): 
		# if next button is there then 
		if next_el != False: 
			# click the next button 
			next_el.click() 
			sleep(2) 
			# comments on the post 
			comment_post(msg)	 
			sleep(2)			 
		else: 
			print("The End") 
			break
		next_el = next_post()

like_till_the_end()
comment_till_the_end()		

```

The program will close after liking/commenting on all posts . Because it cant find the xpath of next on the last post (it will give a run time error).

# To close the browser

This will close the firefox browser which geckodriver is using.

```python

	def quit(self,secs):
		sleep(secs)
		#command to close the browser 
		browser.quit()
		
my_bot.quit(5)

```

**Warnings** If you get the error element not found in xpath (...) , It is because in the future they may update website changing the xpath or its id . A simple fix is to right click on the element in the page and click on inspect . Then right click on the highlighted part -> copy -> xpath , replace the new xpath by overwritting the old one . 

**References** 
* [Youtube](https://www.youtube.com/watch?v=d2GBO_QjRlo)
* [Geeksforgeeks](https://www.geeksforgeeks.org/like-instagram-pictures-using-selenium-python/)