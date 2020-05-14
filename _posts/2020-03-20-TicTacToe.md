---
title: "Python Project : Tic Tac Toe Game"
date: 2020-03-20
tags: [Python]
excerpt: "A fun 2 player game made using python basics"
---

## Tic Tac Toe ( 2 - Player game )

This game is coded using only the basics of Python and a little bit of an extension called colourama .

**Get the required softwares here :**
* [Python](https://www.python.org/)
* [Python Extension - Colorama](https://pypi.org/project/colorama/)

**Download/Clone full code** [here](https://github.com/nvikramraj/TicTacToe)

## To make the game we need to code up a few things :
* The game board
* Getting inputs from the user
* Choosing the winner
* Highlighting the winner 


**Stuff to import before coding**

```python

	import itertools 
	#This is used to switch players after each round.(inbuilt function no need to download)
	from colorama import Fore, Back, Style,init
	init() #initalizes colorama

```

# The game board

This is the code used to get the size of the game map.

```python

    game_size = int(input("What size (size > 1) game of Tic Tac Toe ? ")) 
    # since we are creating n*n size game map , we ask the user for n
    game=[[0 for _ in range(game_size)]for _ in range(game_size)]  
    # creates a n*n array as the game map and gives 0 as default value
    game, _=game_board(game,just_display=True)
    # passes the array to an UDF , to just display the game board

```
The game map is made as a function because it will be called multiple times . In the game map there are two options :
1. To just display the game map (if just_display = True)
2. To change the values in the game map (if just_display = False)

Game board :

![alt]({{ site.url }}{{  site.baseurl  }}/images/TicTacToe/1.jpg)

From the above image you can see that we need some mapping in the game board on the top and size . I have used numbers for mapping because it's easier for the user to interact with.

I have assigned ' X ' as 1 and ' O ' as 2 indicating player 1 has X and player 2 has O
Colorama is used to color X and O as Cyan and Yellow respectively . You can change the color if you want.

A global variable draw is added , It is a draw if there are no empty spaces on the game board and if there is no winner.
 
```python

	def game_board(game_map,player=0,row=0,column=0,just_display=False):

	global draw # this used to check if they draw a match
		
    try:
        
        if game_map[row][column]!=0:
            print("This position is occupied !, Choose another ")
            return game_map,False
        # to check if a player has already played in that position
        
        print("   "+"  ".join([str(i) for i in range(len(game_map))])) 
        # This creates the map on top of the tic tac toe

        if not just_display:
            game_map[row][column] = player
            # To display the game board if just_display is true

        draw = 0

        for count,row in enumerate(game):  # used to change 1 to X and 2 to O and insert colours 
            colored_row=""
            
            for item in row:
                
                if item==0:
                    colored_row +="   "
                    draw+=1 # counting the number of 0's in the list
                # if the index has default value it is left as space
                elif item==1:
                    colored_row+=Fore.CYAN+' X ' + Style.RESET_ALL  
                    #using color Cyan for X
                # if the index has value 1 it is replaced with X
                elif item==2:
                    colored_row+=Fore.YELLOW+' O ' + Style.RESET_ALL 
            		#using color Yellow for O
            	# if the index has value 1 it is replaced with O
            print(count,colored_row)
            # This creates the map on the size of the tic tac toe          
        
        return game_map,True

    # These lines are used for error handling , incase the user enters a number out of range
    except IndexError as e:
        print("Error: Make sure you input row/column between 0 and size -1 ,",e) # modified print statement
        return game_map,False

    except Exception as e:
        print("Something went very wrong! ,",e)	
        return game_map,False

```

# Getting inputs from the user :

We need to get the game size , the index/place on the map they want to play at.
We also need to change the player's turn after each round. (This is done using itertools.cycle and next() an inbuilt function )


**For a more detailed explanation click** [here](https://www.youtube.com/watch?v=eXBD2bB9-RA&list=PLQVvvaa0QuDeAams7fkdcwOGBpGdHpXln)