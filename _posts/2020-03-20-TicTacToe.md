---
title: "Tic Tac Toe"
date: 2020-03-20
tags: [Game]
excerpt: "A fun 2 player game made using python basics"
---

## Tic Tac Toe ( 2 - Player game )

This game is coded using only the basics of Python and a little bit of an extension called colourama .

**Get the required softwares here :**
* [Python](https://www.python.org/)
* [Python Extension - Colorama](https://pypi.org/project/colorama/)

**Download/Clone full code** [here](https://github.com/nvikramraj/TicTacToe)

## To make the game we need to code up a few things :
* The game board.
* Choosing the winner.
* Highlighting the winner.
* Getting inputs from the user .


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
1. To just display the game map (if just_display = True).
2. To change the values in the game map (if just_display = False).

The game needs some mapping in the game board on the top and side . I have used numbers for mapping because it's easier for the user to interact with.

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

        draw =  0 # to count the number of 0s on the board before every round

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

# Choosing the winner :

There are 3 ways to win the game :

1. If a row has the same value (X/O) in every place.
2. If a column has the same value (X/O) in every place.
3. If a diagonal has the same value (X/O) in every place.

This function checks for a winner after every round and if there is a win , It returns True.

```python

def win(current_game):

    def check(lists): #used to check if the list have same elements 
        global player_1,player_2

        if lists.count(lists[0])==len(lists) and lists[0]!=0:
            print(f"Player {lists[0]} is the Winner!") #checks if the no of elements in a list is equal to the size of the list

            if lists[0]==1:  # to keep track of scores 
                player_1+=1
            else:
                player_2+=1

            return True
        else:
            return False

    for count,row in enumerate(current_game): # horizontal winner , check each row
        highlight_row = count
        win_row=[]
        if(check(row)):
            for highlight_col in range(len(row)):
                win_row.append([highlight_row,highlight_col]) #sends the index of the row
            show_winner(current_game,win_row)
            return True

    for col in range(len(current_game)):# Vertical winner , check each column
        vert=[] # vert will store the column elements in a list
        win_col = []
        highlight_col = col
        for highlight_row,row in enumerate(current_game): # check each row the element in col index that is 00 10 20
            vert.append(row[col])
            win_col.append([highlight_row,highlight_col])# sends the index of the column

        if check(vert):
            show_winner(current_game,win_col)
            return True

    dia=[] #diagonal winner , check both diagonals
    win_dia=[]
    for i in range(len(current_game)):
        dia.append(current_game[i][i]) # the index are 00 11 22
        win_dia.append([i,i]) 

    if check(dia):
        show_winner(current_game,win_dia)# send the index of the diagonal
        return True

    dia=[]
    win_dia=[]
    for col,row in enumerate(reversed(range(len(current_game)))):#reversed , gives the row , 2 1 0
        dia.append(current_game[row][col])#enumerate gives the column , that is the index of the list , 0,1,2
        win_dia.append([row,col])

    if check(dia): #the index for this diagonal is 20 11 02
        show_winner(current_game,win_dia) # send the index of the diagonal
        return True

    return False

```

# Highlighting the winner :

To high light the winner , I took the indexes of the row/column/diagonal that won and changed the text background to magenta.

```python

def show_winner(win_game,indexes):   # to highligh the winner 
    #print(indexes)
    print("   "+"  ".join([str(i) for i in range(len(win_game))]))  # to print the column index
    for row_index,row in enumerate(win_game):  
        colored_row=""
        for col_index in range(len(row)):
            #print("a",row_index,col_index)
            for win_index in indexes:  
                #print(win_index[0],win_index[1] , row_index, col_index)
                if  (win_index[0]==row_index)& (win_index[1]==col_index): #To check which indexes won in the game map
                    test=True
                    break
                else :
                    test=False

            if test:
                if (win_game[row_index][col_index]==1):  # highlights only the winning row/column/diagonal
                    colored_row+=Fore.CYAN + Back.MAGENTA+' X ' + Style.RESET_ALL
                elif (win_game[row_index][col_index]==2):
                    colored_row+=Fore.YELLOW + Back.MAGENTA+' O ' + Style.RESET_ALL

            else:
                if (win_game[row_index][col_index]==0): # does not highlight if they lose 
                    colored_row +="   "
                if (win_game[row_index][col_index]==1):
                    colored_row+=Fore.CYAN +' X ' + Style.RESET_ALL
                elif (win_game[row_index][col_index]==2):
                    colored_row+=Fore.YELLOW+' O ' + Style.RESET_ALL    

        print(row_index,colored_row) #prints the game map

```



# Getting inputs from the user :

Finally the user interface.
We need to get the game size , the index/place on the map they want to play at. Make the list that holds the values.

In each round we need to :
1. Switch players.
2. Pass inputs to the game board.
3. Check for win/draw.
4. Ask if they want to play another

If they are done with the game , the program will exit

```python

play = True
player_1 = 0
player_2 = 0
draw = 0
print("Welcome to Tic Tac Toe")

while play:
    print("\nNew Game")
    game_size = int(input("What size (size > 1) game of Tic Tac Toe ? "))  
    game=[[0 for _ in range(game_size)]for _ in range(game_size)]  # creates a game map for the given size 
    game_won=False
    game, _=game_board(game,just_display=True)
    players = itertools.cycle([1,2])  # used to cycle between 2 elements in the list
 
    while not game_won:
        current_player = next(players)
        print(f"\nCurrent Player: {current_player}")
        played=False

        while not played:
            column_choice = int(input("What column do you want to play? ( 0 , 1 , 2 ) : "))
            row_choice = int(input("What row do you want to play? ( 0 , 1 , 2 ) : "))
            game,played = game_board(game, current_player,row_choice,column_choice)
        
        if draw == 0:
            print("\nDraw")
            game_won=True
        else:
            game_won=win(game)

    print(f"\nScores , Player 1 : {player_1} , Player 2 : {player_2} ")
    next_game = input("Do you want to play another match (y/n) : ") 
    play = True if next_game.lower() == "y" else False
print("Byeeeeeeee")

```


**For a more detailed explanation , reference click** [here](https://www.youtube.com/watch?v=eXBD2bB9-RA&list=PLQVvvaa0QuDeAams7fkdcwOGBpGdHpXln)