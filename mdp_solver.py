import numpy as np
import matplotlib.pyplot as plt

# Read file - RN_wrold.txt:
#world_file = 'worlds/assignment_world.txt'
#world_file = 'worlds/assignment_world_b100.txt'
#world_file = 'worlds/assignment_world_prob.txt'
#world_file = 'worlds/assignment_world_p4.txt'
#world_file = 'worlds/assignment_world_discount.txt'
#world_file = 'worlds/hyper_world.txt'
#world_file = 'worlds/labyrinth.txt'
#world_file = 'worlds/labyrinth2.txt'
world_file = 'worlds/RN_world.txt'

world_size_row = 0          #World size X
world_size_column = 0       #World size Ys

start_point_row = 0         #Starting point coordiantes X
start_point_column =0       #Starting point coordiantes Y

p1 = 0                      #Probability of going to intended state
p2 = 0                      #Probability of going left of the action's origin
p3 = 0                      #Probability of going right of the action's origin
p4 = 0                      #Probability of going opposite of the intended state 

reward = 0                  #Reward/cost of move

discounting_factor = 0      #Discounting factor

terminal_state_list = []    #List of terminal states
prohibited_state_list = []  #List of prohibited states
special_state_list =[]      #List of special states

exploration_param = 0       #Exploration parameter

with open(world_file,'r') as f:

    # Setting parameters of world, read from file
    for lines in f:
        line = lines.split()
        
        if line[0] == 'W':
            world_size_column = int(line[1])
            world_size_row = int(line[2])
            
        
        if line[0] == 'S':
            start_point_column = int(line[1]) - 1
            start_point_row = int(line[2]) -1
        
        if line[0] == 'P':
            p1 = float(line[1])
            p2 = float(line[2])
            p3 = float(line[3])
            p4 = 1 - float(p1) - float(p2) - float(p3)
            if p4 <= 0:
                p4 = 0

        if line[0] == 'R':
            reward = float(line[1])

        if line[0] == 'G':
            discounting_factor = float(line[1])

        if line[0] == 'T':
            terminal_state_column = int(line[1]) - 1
            terminal_state_row =  int(line[2]) - 1
            terminal_state_value =  float(line[3])
            terminal_state_point = [terminal_state_column, terminal_state_row, terminal_state_value]
            terminal_state_list.append(terminal_state_point)
        
        if line[0] == 'F':
            prohibited_state_column =  int(line[1]) - 1
            prohibited_state_row =  int(line[2]) - 1
            prohibited_state_point = [prohibited_state_column, prohibited_state_row]
            prohibited_state_list.append(prohibited_state_point)
        
        if line[0] == 'B':
            special_state_column =  int(line[1]) - 1
            special_state_row =  int(line[2]) - 1
            special_state_value =  float(line[3])
            special_state_point = [special_state_column, special_state_row, special_state_value]
            special_state_list.append(special_state_point)

        if line[0] == 'E':
            exploration_param = line[1]


print('W World size:                          ' + str(world_size_column) + ' x ' + str(world_size_row))
print('S Starting point:                      ' + str(start_point_column + 1) + ' x ' + str(start_point_row + 1))
print('P Probabilities(p1, p2, p3, p4):       ' + str(p1) + ', ' + str(p2) + ', ' + str(p3) + ', ' + str(p4))
print('R Reward:                              ' + str(reward))
print('G Discounting factor:                  ' + str(discounting_factor))

for ts in terminal_state_list:
    print('T Terminal state:                      ' + str(ts[0]+1) + ", " + str(ts[1]+1) + ", " + str(ts[2]))

for ps in prohibited_state_list:
    print('F Prohibited states:                   ' + str(ps[0]+1) + ", " + str(ps[1]+1))

for ss in special_state_list:
    print('B Special states:                      ' + str(ss[0]+1) + ", " + str(ss[1]+1) + ", " + str(ss[2]))

print('E Exploration parameter:               ' + str(exploration_param))
print()

#Drawing map based on the parameters
def draw_map(column, row, spy, spx, tsl, psl, ssl):
    for j in range(column):
        print("+---", end="")
    print("+")
    
    for i in reversed(range(row)):
        for j in range(column):
            flag_T = 0
            flag_F = 0
            flag_B = 0
            for T in tsl:
                Ty = T[0]
                Tx = T[1]
                if j == Ty and i == Tx:
                    print("| T ", end="")
                    flag_T = 1
                
            for B in ssl:
                By = B[0]
                Bx = B[1]
                if j == By and i == Bx:
                    print("| B ", end="")
                    flag_B = 1
            
            for F in psl:
                Fy = F[0]
                Fx = F[1]
                if j == Fy and i == Fx:
                    print("| F ", end="")
                    flag_F = 1

            if j == spy and i == spx:
                print("| S ", end="")
            else:
                if flag_T == 0 and flag_F == 0 and flag_B == 0:
                    print("|   ", end="")

        print("|")
        
        for j in range(column):
            print("+---", end="")
        print("+")

draw_map(world_size_column, world_size_row, 
         start_point_column, start_point_row,
         terminal_state_list,
         prohibited_state_list,
         special_state_list)
print()



# Step 2: Initialize the value function
V = np.zeros((world_size_column, world_size_row))

# Set the value of terminal states
for ts in terminal_state_list:
    ts_column, ts_row, ts_value = ts
    V[ts_column][ts_row] = ts_value

# Set the value of prohibited states
for ps in prohibited_state_list:
    ps_column, ps_row = ps
    ps_value = 0
    V[ps_column][ps_row] = ps_value

# Set the value of special states
for ss in special_state_list:
    ss_column, ss_row, ss_value = ss
    V[ss_column][ss_row] = ss_value


# Print the initialized value function
print("Initialized Value Function:")
V_print = np.flip(np.transpose(V), axis=0)
print(V_print)
print()

# Initialize an empty list to store utility values for analysis
utility_values = []
utility_values.append(np.copy(V))

# Step 3: Perform value iteration
epsilon = 0.0001  # Convergence threshold
iterations = 10000
it=0
while True:
    delta = 0  # Initialize delta for convergence check
    
    # Create a copy of the value function for comparison
    V_prev = np.copy(V)
    # print(V_prev)
    # print()

    # Update the value function for each state
    for i in range(world_size_row):
        for j in range(world_size_column):
            
            # Skip terminal and prohibited states
            if [j, i] in prohibited_state_list:
                #print('skipped prohibitted state')
                continue

            flag_skippded_ts = 0
            for ts in terminal_state_list:
                if j == ts[0] and i == ts[1]:
                    flag_skippded_ts = 1
            if flag_skippded_ts == 1:
                #print('skipped terminal state')
                continue

            flag_skippded_ss = 0
            for ss in special_state_list:
                if j == ss[0] and i == ss[1]:
                    flag_skippded_ss = 1
            if flag_skippded_ss == 1:
                #print('skipped special state')
                continue
            
            # Calculate the expected value for each action
            if j - 1 < 0 and i - 1 < 0:
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]


                V_up = p1 * V_prev[j][i+1] + p2 * V_prev[j][i] + p3 * V_prev[j+1][i] + p4 * V_prev[j][i]
                V_down = p1 * V_prev[j][i] + p2 * V_prev[j+1][i] + p3 * V_prev[j][i] + p4 * V_prev[j][i+1] 
                V_left = p1 * V_prev[j][i] + p2 * V_prev[j][i] + p3 * V_prev[j][i+1] + p4 * V_prev[j+1][i] 
                V_right = p1 * V_prev[j+1][i] + p2 * V_prev[j][i+1] + p3 * V_prev[j][i] + p4 * V_prev[j][i]



            elif j - 1 < 0 and i + 1 >= world_size_row:
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]

                V_up = p1 * V_prev[j][i] + p2 * V_prev[j][i] + p3 * V_prev[j+1][i] + p4 * V_prev[j][i-1]
                V_down = p1 * V_prev[j][i-1] + p2 * V_prev[j+1][i] + p3 * V_prev[j][i] + p4 * V_prev[j][i] 
                V_left = p1 * V_prev[j][i] + p2 * V_prev[j][i-1] + p3 * V_prev[j][i] + p4 * V_prev[j+1][i] 
                V_right = p1 * V_prev[j+1][i] + p2 * V_prev[j][i] + p3 * V_prev[j][i-1] + p4 * V_prev[j][i]

            
            elif j - 1 < 0 :
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]

                V_up = p1 * V_prev[j][i+1] + p2 * V_prev[j][i] + p3 * V_prev[j+1][i] + p4 * V_prev[j][i-1]
                V_down = p1 * V_prev[j][i-1] + p2 * V_prev[j+1][i] + p3 * V_prev[j][i] + p4 * V_prev[j][i+1] 
                V_left = p1 * V_prev[j][i] + p2 * V_prev[j][i-1] + p3 * V_prev[j][i+1] + p4 * V_prev[j+1][i] 
                V_right = p1 * V_prev[j+1][i] + p2 * V_prev[j][i+1] + p3 * V_prev[j][i-1] + p4 * V_prev[j][i]



            elif j+1 >= world_size_column and i - 1 < 0:
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]

                V_up = p1 * V_prev[j][i+1] + p2 * V_prev[j-1][i] + p3 * V_prev[j][i] + p4 * V_prev[j][i]
                V_down = p1 * V_prev[j][i] + p2 * V_prev[j][i] + p3 * V_prev[j-1][i] + p4 * V_prev[j][i+1] 
                V_left = p1 * V_prev[j-1][i] + p2 * V_prev[j][i] + p3 * V_prev[j][i+1] + p4 * V_prev[j][i] 
                V_right = p1 * V_prev[j][i] + p2 * V_prev[j][i+1] + p3 * V_prev[j][i] + p4 * V_prev[j-1][i]



            elif j+1 >= world_size_column and i + 1 >= world_size_row:
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]

                V_up = p1 * V_prev[j][i] + p2 * V_prev[j-1][i] + p3 * V_prev[j][i] + p4 * V_prev[j][i-1]
                V_down = p1 * V_prev[j][i-1] + p2 * V_prev[j][i] + p3 * V_prev[j-1][i] + p4 * V_prev[j][i] 
                V_left = p1 * V_prev[j-1][i] + p2 * V_prev[j][i-1] + p3 * V_prev[j][i] + p4 * V_prev[j][i] 
                V_right = p1 * V_prev[j][i] + p2 * V_prev[j][i] + p3 * V_prev[j][i-1] + p4 * V_prev[j-1][i]


    
            elif j+1 >= world_size_column :
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]

                V_up = p1 * V_prev[j][i+1] + p2 * V_prev[j-1][i] + p3 * V_prev[j][i] + p4 * V_prev[j][i-1]
                V_down = p1 * V_prev[j][i-1] + p2 * V_prev[j][i] + p3 * V_prev[j-1][i] + p4 * V_prev[j][i+1] 
                V_left = p1 * V_prev[j-1][i] + p2 * V_prev[j][i-1] + p3 * V_prev[j][i+1] + p4 * V_prev[j][i] 
                V_right = p1 * V_prev[j][i] + p2 * V_prev[j][i+1] + p3 * V_prev[j][i-1] + p4 * V_prev[j-1][i]
    


            elif i+1 >= world_size_row :
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]

                V_up = p1 * V_prev[j][i] + p2 * V_prev[j-1][i] + p3 * V_prev[j+1][i] + p4 * V_prev[j][i-1]
                V_down = p1 * V_prev[j][i-1] + p2 * V_prev[j+1][i] + p3 * V_prev[j-1][i] + p4 * V_prev[j][i] 
                V_left = p1 * V_prev[j-1][i] + p2 * V_prev[j][i-1] + p3 * V_prev[j][i] + p4 * V_prev[j+1][i] 
                V_right = p1 * V_prev[j+1][i] + p2 * V_prev[j][i] + p3 * V_prev[j][i-1] + p4 * V_prev[j-1][i]


            elif i - 1 < 0:
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]

                V_up = p1 * V_prev[j][i+1] + p2 * V_prev[j-1][i] + p3 * V_prev[j+1][i] + p4 * V_prev[j][i]
                V_down = p1 * V_prev[j][i] + p2 * V_prev[j+1][i] + p3 * V_prev[j-1][i] + p4 * V_prev[j][i+1] 
                V_left = p1 * V_prev[j-1][i] + p2 * V_prev[j][i] + p3 * V_prev[j][i+1] + p4 * V_prev[j+1][i] 
                V_right = p1 * V_prev[j+1][i] + p2 * V_prev[j][i+1] + p3 * V_prev[j][i] + p4 * V_prev[j-1][i]


            else:
                if [j-1, i] in prohibited_state_list:
                    V_prev[j-1][i] = V_prev[j][i]

                elif [j+1, i] in prohibited_state_list:
                    V_prev[j+1][i] = V_prev[j][i]

                elif [j, i+1] in prohibited_state_list:
                    V_prev[j][i+1] = V_prev[j][i]
                
                elif [j, i-1] in prohibited_state_list:
                    V_prev[j][i-1] = V_prev[j][i]
                    
                V_up = p1 * V_prev[j][i+1] + p2 * V_prev[j-1][i] + p3 * V_prev[j+1][i] + p4 * V_prev[j][i-1]
                V_down = p1 * V_prev[j][i-1] + p2 * V_prev[j+1][i] + p3 * V_prev[j-1][i] + p4 * V_prev[j][i+1] 
                V_left = p1 * V_prev[j-1][i] + p2 * V_prev[j][i-1] + p3 * V_prev[j][i+1] + p4 * V_prev[j+1][i] 
                V_right = p1 * V_prev[j+1][i] + p2 * V_prev[j][i+1] + p3 * V_prev[j][i-1] + p4 * V_prev[j-1][i]


            # Calculate the maximum expected value and update the value function
            max_value = max(V_up, V_down, V_left, V_right)
            new_value = reward + discounting_factor * max_value
            delta = max(delta, abs(new_value - V_prev[j][i]))  # Update delta for convergence check
            V[j][i] = new_value
            V_prev[j][i] = new_value
        
    # Save the current utility values for analysis
    utility_values.append(np.copy(V))
    
    it=it+1
    
    # Check for convergence
    if delta < epsilon:
        #print("Delta break")
        break

    
    if it >= iterations:
        #print("Iteration break")
        break

# Print the final value function after value iteration
# print()
print("Final Value Function:")
#print(V)
#print()

V_print = np.flip(np.transpose(V), axis=0)
np.set_printoptions(precision=4, suppress=True)
print(V_print)
print()


#Step 4: Determine the optimal policy based on the value function
optimal_policy = np.zeros((world_size_column, world_size_row), dtype=str)

for j in range(world_size_column):
    for i in range(world_size_row):

        #Skip prohibited states
        if [j, i] in prohibited_state_list:
            optimal_policy[j,i] = 'F'
            #print('skipped prohibitted state')
            continue
        
        #Skip terminal and terminal states
        flag_skippded_ts = 0
        for ts in terminal_state_list:
            if j == ts[0] and i == ts[1]:
                flag_skippded_ts = 1
                optimal_policy[j,i] = 'T'
                #print(optimal_policy[j,i])
        if flag_skippded_ts == 1:
            #print('skipped terminal state')
            continue
        
        #Skip special prohibited states
        flag_skippded_ss = 0
        for ss in special_state_list:
            if j == ss[0] and i == ss[1]:
                flag_skippded_ss = 1
                optimal_policy[j,i] = 'B'
                #print(optimal_policy[j,i])
        if flag_skippded_ss == 1:
            #print('skipped specialstate')
            continue

        #Calculate the expected value for each action
        V_up = V_down = V_left = V_right = 0.0

        #Check boundaries for up action
        if i + 1 < world_size_row:
            V_up = p1 * V[j][i+1]
        else:
            V_up = p1 * V[j][i]
        
        #Check boundaries for down action
        if i - 1 >= 0:
            V_down = p1 * V[j][i-1]
        else:
            V_down = p1 * V[j][i]
        
        #Check boundaries for left action
        if j - 1 >= 0:
            V_left = p1 * V[j-1][i]
        else:
            V_left = p1 * V[j][i]
        
        #Check boundaries for right action
        if j + 1 < world_size_column:
            V_right = p1 * V[j+1][i]
        else:
            V_right = p1 * V[j][i]
        
        #Find the action that leads to the maximum expected value
        max_value = max(V_up, V_down, V_left, V_right)
        
        if max_value == V_up:
            optimal_policy[j][i] = '^'
        elif max_value == V_down:
            optimal_policy[j][i] = 'v'
        elif max_value == V_left:
            optimal_policy[j][i] = '<'
        elif max_value == V_right:
            optimal_policy[j][i] = '>'

#Print the optimal policy
print("Optimal Policy:")
optimal_policy = np.flip(np.transpose(optimal_policy), axis=0)
print(optimal_policy)
print()

# Convert the utility_values list to a NumPy array for easier plotting
utility_values = np.array(utility_values)

# Plot the utility values
plt.figure()
for i in range(world_size_row):
    for j in range(world_size_column):
        plt.plot(range(it + 1), utility_values[:, j, i], label=f'State ({j+1}, {i+1})')

plt.xlabel('Iterations')
plt.ylabel('Utility')
plt.legend(loc='lower right')
#plt.legend(loc='best')
plt.title('Convergence of Utility Values')
plt.grid(True)
plt.show()