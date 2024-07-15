import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import model as model

class TeamBuilderApp(tk.Frame): # creates drop down menu of teams to select players
	def __init__(self, root, x, layer):
		tk.Frame.__init__(self, root)
		self.column = []
		self.layer = layer + 1
		self.root = root
		self.confirm = tk.Button(self.root, text = "confirm", font=('arial', 12), background= buttonColour, fg= fontColour, borderwidth=0, highlightthickness=0 ,command= lambda: self.addTeam(self.selected.get()))
		self.addTeamButton = tk.Button(self.root, text = 'Add Team', background= buttonColour, fg = fontColour, borderwidth=0, highlightthickness=0, command= lambda : self.addCollumn())
		self.selected = tk.StringVar()
		self.selected.set(teams[0])
		self.teamSelect = tk.OptionMenu(self.root, self.selected, *teams)
		self.teamSelect.configure(bg= buttonColour, fg = fontColour, borderwidth=0, highlightthickness=0)
		self.x = x
		self.players = []
		self.chosen = []
		self.abr = ""
		self.confirm.place(x = self.x, y=130)
		self.teamSelect.place(x =self.x, y = 90)
		self.playerButtons = []


	def addOption(self): # adds the option for another team column to next column
		self.addTeamButton.place(x= self.x + 200, y = 90)

	def addCollumn(self): # adds another team column if selected
		next = self.x + 200
		new = TeamBuilderApp(self.root, self.x + 200, self.layer)
		if self.layer < 5:
			new.addOption()
		self.addTeamButton.destroy()
		self.column.append(new)
	

	def addPlayer(self, y, player): # adds a player button to GUI
		button = tk.Button(self.root, text=player, width= 20, font=('Arial', 12), background= buttonColour, borderwidth=0, highlightthickness=0, fg = fontColour)
		button.configure(command= lambda b = button: self.clickPlayer(b))
		button.place(y=y, x=self.x)
		self.playerButtons.append(button)
		self.players.append(button)

	def addTeam(self, team): # adds player buttons when team is selected

		team_row = current_teams[current_teams['TEAM'] == team] # removes player buttons if they were already there
		if team_row["ABR"].values[0] != self.abr:
			self.chosen.clear()
			for label in self.playerButtons:
				label.destroy()

		self.abr = team_row["ABR"].values[0]
		players = current_players[current_players["TEAM"] == self.abr]["PLAYER"].values
		addY = 0
		for player in players:
			self.addPlayer(160 + addY, player)
			addY += 23

	def clearPlayers(self): # clears player buttons in a column
		for button in self.players:
			button.destroy()

	def clickPlayer(self, button): # changes colour of player and adjusts selected lists accordingly

		if(button['bg'] == '#545663'): # remove player from selected
			button.configure(bg= buttonColour)
			self.chosen.remove(button["text"])
		else:
			button.configure(bg = "#545663")
			self.chosen.append(button["text"]) # add player from selected

current_teams = model.team_df[model.team_df["Year"] == model.current_year - 1]
current_players = model.players_df[model.players_df["Year"] == model.current_year - 1] # loads in all players and teams from the dfs inmodel

bgColour = "#202123"
buttonColour = "#353541"
fontColour = "#C5C5D1"
mode = ["Neutral", "Offensive", "Defensive"]


team_names_array = current_teams['TEAM'].unique()

# If you want to convert the result to a regular Python list, you can use the tolist() method
teams = team_names_array.tolist()

# Print or use the result as needed


class tkinterApp(tk.Tk):
	
	# __init__ function for class tkinterApp 
	def __init__(self, *args, **kwargs): 
		
		# __init__ function for class Tk
		tk.Tk.__init__(self, *args, **kwargs)
		
		# creating a container
		container = tk.Frame(self) 
		container.pack(side = "top", fill = "both", expand = True) 

		container.grid_rowconfigure(0, weight = 1)
		container.grid_columnconfigure(0, weight = 1)

		# initializing frames to an empty array
		self.frames = {} 

		# iterating through a tuple consisting
		# of the different page layouts
		for F in (StartPage, Evaluate, Recommend):

			frame = F(container, self)

			# initializing frame of that object from
			# startpage, page1, page2 respectively with 
			# for loop
			self.frames[F] = frame 

			frame.grid(row = 0, column = 0, sticky ="nsew")

		self.show_frame(StartPage)

	# to display the current frame passed as
	# parameter
	def show_frame(self, cont):
		frame = self.frames[cont]
		frame.tkraise()

# first window frame startpage

class StartPage(tk.Frame):
	def __init__(self, parent, controller): 
		# Initialize the frame with parent as its parent widget
		tk.Frame.__init__(self, parent)

		# Set background color of the frame
		self.configure(background=bgColour)

		# Label for home title with specified text, font, colors, and no border
		homeTitle = tk.Label(self, text='Select An Option', font=('Arial', 20), background=bgColour, fg=fontColour, borderwidth=0, highlightthickness=0)

		# Button to navigate to the Evaluate page with specified text, font, colors, and command
		evalButton = tk.Button(self, text='Evaluate', font=('Arial', 20), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0, command=lambda: controller.show_frame(Evaluate))

		# Button to navigate to the Recommend page with specified text, font, colors, and command
		recomButton = tk.Button(self, text='Recommend', font=('Arial', 20), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0, command=lambda: controller.show_frame(Recommend))

		# Label for description of the Evaluate button with specified text, colors, and no border
		evalDesc = tk.Label(self, text='Evaluates an input trade', background=bgColour, fg=fontColour, borderwidth=0, highlightthickness=0)

		# Label for description of the Recommend button with specified text, colors, and no border
		recomDesc = tk.Label(self, text='Has the AI recommend trades', background=bgColour, fg=fontColour, borderwidth=0, highlightthickness=0)

		# Place the home title label with specified padding
		homeTitle.pack(padx=0, pady=20)

		# Place the evaluate button at specified position
		evalButton.place(x=375, y=200)

		# Place the recommend button at specified position
		recomButton.place(x=750, y=200)

		# Place the evaluate description label at specified position
		evalDesc.place(x=372, y=270)

		# Place the recommend description label at specified position
		recomDesc.place(x=760, y=270)


		


# second window frame evaluate 
class Evaluate(tk.Frame):

	
	def __init__(self, parent, controller):
		
		# Initialize the frame with parent as its parent widget
		tk.Frame.__init__(self, parent)

		# Label to display prediction, with specified text, font, colors, and position
		label_result = tk.Label(self, text="Prediction: N Wins", font=("Arial", 14), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0)
		label_result.place(x=1100, y=30)

		# Label to display win difference, with specified text, font, colors, and position
		winDiff = tk.Label(self, text="Win Difference: X", font=("Arial", 14), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0)
		winDiff.place(x=1100, y=60)

		# Initialize an empty list to hold player labels
		self.playerLabels = []

		# Set background color of the frame
		self.config(background=bgColour)

		# Create an instance of TeamBuilderApp with specified parameters
		teamColumns = TeamBuilderApp(self, 20, 0)

		# Add an option to the team columns
		teamColumns.addOption()

		# Button to navigate back to the StartPage, with specified text, colors, and command
		backButton = tk.Button(self, text='Back', background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0, command=lambda: controller.show_frame(StartPage))
		backButton.place(x=0, y=0)

		# Button to evaluate, with specified text, font, colors, and command
		evalButton = tk.Button(self, text='Evaluate', font=('Arial', 12), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0, command=lambda: calculateValue(label_result, teamColumns, winDiff))
		evalButton.place(x=600, y=20)

		# Button to open tutorial, with specified text, font, colors, and command
		tutorialButton = tk.Button(self, text='Tutorial', font=('Arial', 12), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0, command=lambda: open_eval_tutorial(self))
		tutorialButton.place(x=600, y=50)

		# Place the back button at specified position
		backButton.place(x=0, y=0)





# third window frame page2
class Recommend(tk.Frame): 

	def __init__(self, parent, controller):

		# Initialize the frame with parent as its parent widget
		tk.Frame.__init__(self, parent)

		# Set background color of the frame
		self.configure(background=bgColour)

		# Initialize an empty list to hold player labels
		self.playerLabels = []

		# Label to display prediction, with specified text, font, colors, and position
		label_result = tk.Label(self, text="Prediction: N Wins", font=("Arial", 14), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0)
		label_result.place(x=1100, y=30)

		# Label to display win difference, with specified text, font, colors, and position
		winDiff = tk.Label(self, text="Win Difference: X", font=("Arial", 14), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0)
		winDiff.place(x=1100, y=60)

		# Create a horizontal slider using TeamBuilderApp
		column = TeamBuilderApp(self, 20, 0)

		# Button to navigate back to the StartPage, with specified text, font, colors, and command
		backButton = tk.Button(self, text='Back', font=('Arial', 12), background=buttonColour, borderwidth=0, highlightthickness=0, fg=fontColour, command=lambda: controller.show_frame(StartPage))

		# Button to get recommendations, with specified text, font, colors, and command
		recomButton = tk.Button(self, text='Recommend', font=('Arial', 12), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0, command=lambda: get_recommendation(self, column, label_result, winDiff))

		# Button to open recommendation tutorial, with specified text, font, colors, and command
		tutorialButton = tk.Button(self, text='Tutorial', font=('Arial', 12), background=buttonColour, fg=fontColour, borderwidth=0, highlightthickness=0, command=lambda: open_recom_tutorial(self))

		# Place the recommend button at specified position
		recomButton.place(x=600, y=20)

		# Place the tutorial button at specified position
		tutorialButton.place(x=600, y=50)

		# Place the back button at specified position
		backButton.place(x=0, y=0)


def calculateValue(result_label, columns, winDiff):
	# caclulates th win difference and number wins based on a trade result
	teamPlayers = columns.chosen
	acquiredPlayers = []
	temp = columns
	while(len(temp.column) > 0):
		temp = temp.column[0]
		acquiredPlayers += temp.chosen
	result = model.predict_trade(acquiredPlayers, teamPlayers, columns.abr)
	evaluation = result + model.team_df[(model.team_df["Year"] == model.current_year - 1) & (model.team_df["ABR"] == columns.abr)]["W"].values[0]
	evaluation = round(evaluation, 2)
	result = round(result, 2)
	result_label.config(text=f"Preditcion: {evaluation:.2f} wins")
	winDiff.config(text=f"Win Difference: {result:.2f}")

def get_recommendation(self, column, result_label, winDiff):
	# queries the model for trade recommendation and out puts it
	if column.abr == "": # if no team selected
		return 0
	result = model.recommend(column.abr, column.chosen)
	IN_label = tk.Label(self, text="PLAYER(S) IN", font=("Arial", 14), background= buttonColour, fg = fontColour, borderwidth=0, highlightthickness=0)
	IN_label.place(x = 420, y = 130)
	Out_label = tk.Label(self, text="PLAYER(S) OUT", font=("Arial", 14), background= buttonColour, fg = fontColour, borderwidth=0, highlightthickness=0)
	Out_label.place(x = 220, y = 130)
	Inlabels = []
	outLabels = []
	nextY = 160
	for playerIn in result[0]:
		print(playerIn)
		new = tk.Label(self, text=playerIn, font=("Arial", 12), background= buttonColour, fg = fontColour, borderwidth=0, highlightthickness=0)
		new.place(x = 420, y =nextY)
		nextY += 23
		Inlabels.append(new)
	nextY = 160
	for playerOut in result[1]:
		print(playerOut)
		new = tk.Label(self, text=playerOut, font=("Arial", 12), background= buttonColour, fg = fontColour, borderwidth=0, highlightthickness=0)
		new.place(x = 220, y =nextY)
		nextY += 23
		outLabels.append(new)
	evaluation = model.team_df[(model.team_df["Year"] == model.current_year - 1) & (model.team_df["ABR"] == column.abr)]["W"].values[0]
	evaluation += result[2]
	evaluation = round(evaluation, 2)
	result_label.config(text=f"Prediction {evaluation:.2f} wins")
	winDiff.config(text=f"Win Difference: {result[2]:.2f}")

def open_eval_tutorial(self):
	tutorial_window = tk.Toplevel(self)
	tutorial_window.title("Tutorial")

	# Add explanatory text using a scrolled text widget
	tutorial_text = """
	Tutorials for Evaluation window

	Here is a brief tutorial on how to use this program:
	1. On the left most column you should select the team that you want an evaluation of
	2. Click the players on your team you are trading away
	3. Then use the add teams buttons to add more columns of which you can select teams in
	4. Select all the players from those teams you are getting in return for your trade
	5. Click the Evaluate button and see the results outputted as a predicted number of wins
	for your team next season as well as the difference in wins that would be compared to your
	most recent season

	Feel free to explore and enjoy using the application!
	"""

	text_widget = scrolledtext.ScrolledText(tutorial_window, wrap=tk.WORD, width=100, height=15)
	text_widget.insert(tk.INSERT, tutorial_text)
	text_widget.pack(padx=10, pady=10)

def open_recom_tutorial(self):
	tutorial_window = tk.Toplevel(self)
	tutorial_window.title("Tutorial")

	# Add explanatory text using a scrolled text widget
	tutorial_text = """
	Tutorials for Recommend window

	Here is a brief tutorial on how to use this program:
	1. On the left most column you should select the team that you want an evaluation of
	2. Click the players you want to void from being traded
	3. Select the trade type you would like in the option menu containing neutral. Choose based
	on what you want the trade to predict, offense or defense
	4. Press the Recommend Button and see the results outputted as a predicted number of wins
	for your team next season as well as the difference in wins that would be compared to your
	most recent season

	Feel free to explore and enjoy using the application!
	"""

	text_widget = scrolledtext.ScrolledText(tutorial_window, wrap=tk.WORD, width=100, height=15)
	text_widget.insert(tk.INSERT, tutorial_text)
	text_widget.pack(padx=10, pady=10)

# Driver Code
app = tkinterApp()
app.configure(background=bgColour)
app.geometry("2000x650")
app.mainloop()
