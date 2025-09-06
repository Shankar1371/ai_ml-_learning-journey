#basic python data types are
x=10 #integer
pi=3.14 #float
name="AI" #String
is_ready = True #boolean

print(type(is_ready), type(pi), type(name), type(x))


#Lists they are mutable
numbers =[1,2,3,4,5]
print("First element: ", numbers[0])
print("slice:", numbers[1:4])

#tuples they are immutable that cannot be changed after they have been created
coords = (10, 20)
print("Tuple: ", coords)

#Dictionaries   these are like hashmap in java that has key value pairs
student = { "name": "shankar", "age": 23, "major" : "CS"}
print("keys:",student.keys()) #this gets the student keys
print("Name:", student.get("name")) #this gets the name by using the get function

#sets
#sets are the built in data type used to store collection of unique and immutable elements.
# they are unordered meaning  the elements which means they can be kept anywhere
#they can store all the collection of datasets
unique_nums= {1,2,2,3,4,5}
print("unique sets:", unique_nums)


