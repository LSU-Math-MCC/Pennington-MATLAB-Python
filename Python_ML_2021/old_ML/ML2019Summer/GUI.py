from tkinter import *


#this is our Master Widget we will probably add everything onto. It inherits from the frame class in tkinter

class Window(Frame):


    def __init__(self, master=None):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

        self.greet_button = Button(master, text="Run", command=self.greet)
        self.greet_button.pack()



    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("IDK if this is even worth it")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        quitButton = Button(self, text="End My Misery",command=self.client_exit , bg="orange", fg="red")

        # placing the button on my window. the axis open
        quitButton.place(x=700, y=550)



    #method to define quitting the windows
    def client_exit(self):
        exit()

    def greet(self):
        print("This Program is absolute shit!")


# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

#size of the window
root.geometry("800x600")
#creating the Instance
app = Window(root)
#this is the run the main loop that waits on the dialog box input
root.mainloop()
