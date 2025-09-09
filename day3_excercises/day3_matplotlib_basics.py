import matplotlib.pyplot as plt

#line plot
x =[1,2,3,4,5]
y = [2,4,6,8,10]
plt.plot(x,y, marker='o',linestyle='-',color='blue')
#here we have plot this two dimensional data that is x and y  and the marker, linestyle and color is set
plt.title("simple Line plot")
#we have the function title that sets the title for the graph that is created
plt.xlabel("X values")
plt.ylabel("Y values")
plt.grid(True)
plt.show()

#bar chart
categories= ["apples", "bananas", "pineapple"]
values =[10,15,7]
plt.bar(categories, values,color=['red','blue','green'])
#here we have the bar chart that has been created by categories and values that are set
#and we also set the colors
plt.title("bar plot")
plt.show()

#and this is the matplotlib basics that i have learnt