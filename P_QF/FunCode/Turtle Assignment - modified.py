import turtle
import time
turtle.setup(width=800, height=800)  # This is the size of the canvas
turtle.title("Cartesian Coordinate.")  # This is the title of the canvas
value = 'STRING'
widthA = None

def get_mouse_click_coor(x, y):
    turtle.onscreenclick(None)
    print(x, y)

'''for _ in range(4):   # 4 times makes a square
  turtle.fd(100)  #turtle.forward
  turtle.lt(90)   #turtle.left 90 deg '''

screen = turtle.Screen()
screen.title("Cartesian Coordinate")
screen.setup(width=800, height=800)

# Create a turtle object
t = turtle.Turtle()
t.speed(0)  # Set the drawing speed to the fastest

# Draw X and Y axes
t.penup()
t.goto(-400, 0)
t.pendown()
t.goto(400, 0)
t.penup()
t.goto(0, -400)
t.pendown()
t.goto(0, 400)


widthA = turtle.numinput('Cartesian Coordinate', 'Enter hidth [800]', minval=100, maxval=800) # This is the dialog box
while widthA < 100:
    print('The allowed minimum value is 100.')
    continue
while widthA > 800:
    print('The allowed maxium value is 800.')
    continue

heightA = turtle.numinput('Cartesian Coordinate', 'Enter height [600]', minval=100, maxval=800) # This is the dialog box

tickA = turtle.numinput('Cartesian Coordinate', 'Enter tick spacing (10, 20, 50, or 100) [50]', minval=10, maxval=100) # This is the dialog box


# Label the X and Y axes
t.penup()
t.color("blue")
for x in range(-400, 400, 50):
    t.goto(x, -300)
    t.write(str(x), align="left")
for y in range(-400, 400, 50):
    t.goto(-300, y)
    t.write(str(y), align="right")
    
print(widthA, heightA, tickA)
time.sleep(20)  # Wait for 1 second before drawing the text
t.penup()
t.color("red")
turtle.write('hello world ' + str(value), align='center', font=('Arial', 20, 'normal'))  #This writes text to the canvas
time.sleep(20)  # Wait for 1 second before closing the window
turtle.onscreenclick(get_mouse_click_coor)
