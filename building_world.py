# Read file - RN_wrold.txt:
world_file = 'worlds/RN_world.txt'
#world_file = 'worlds/assignment_world.txt'

world_size_row = 0          #World size X
world_size_column = 0       #World size Y

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
            world_size_row = int(line[1])
            world_size_column = int(line[2])
        
        if line[0] == 'S':
            start_point_row = int(line[1]) - 1
            start_point_column = int(line[2]) -1
        
        if line[0] == 'P':
            p1 = line[1]
            p2 = line[2]
            p3 = line[3]
            p4 = 1 - float(p1) - float(p2) - float(p3)
            if p4 <= 0:
                p4 = 0

        if line[0] == 'R':
            reward = line[1]

        if line[0] == 'G':
            discounting_factor = line[1]

        if line[0] == 'T':
            terminal_state_row = int(line[1]) - 1
            terminal_state_column =  int(line[2]) - 1
            terminal_state_value =  int(line[3])
            terminal_state_point = [terminal_state_row, terminal_state_column, terminal_state_value]
            terminal_state_list.append(terminal_state_point)
        
        if line[0] == 'F':
            prohibited_state_row =  int(line[1]) - 1
            prohibited_state_column =  int(line[2]) - 1
            prohibited_state_point = [prohibited_state_row, prohibited_state_column]
            prohibited_state_list.append(prohibited_state_point)
        
        if line[0] == 'B':
            special_state_row =  int(line[1]) - 1
            special_state_column =  int(line[2]) - 1
            special_state_value =  int(line[3])
            special_state_point = [special_state_row, special_state_column, special_state_value]
            special_state_list.append(special_state_point)

        if line[0] == 'E':
            exploration_param = line[1]


print('World size:                          ' + str(world_size_row) + ' x ' + str(world_size_column))
print('Starting point:                      ' + str(start_point_row) + ' x ' + str(start_point_column))
print('Probabilities(p1, p2, p3, p4):       ' + str(p1) + ', ' + str(p2) + ', ' + str(p3) + ', ' + str(p4))
print('Reward:                              ' + str(reward))
print('Discounting factor:                  ' + str(discounting_factor))

for ts in terminal_state_list:
    print('Terminal state:                      ' + str(ts))

for ps in prohibited_state_list:
    print('Prohibited states:                   ' + str(ps))

for ss in special_state_list:
    print('Special states:                      ' + str(ss))

print('Exploration parameter:               ' + str(exploration_param))
print()

#Drawing map based on the parameters
def draw_map(x, y, spx, spy, tsl, psl, ssl):
    for j in range(x):
        print("+---", end="")
    print("+")
    
    for i in reversed(range(y)):
        # flag_T = 0
        # flag_F = 0
        for j in range(x):
            flag_T = 0
            flag_F = 0
            flag_B = 0
            for T in tsl:
                Tx = T[0]
                Ty = T[1]
                if j == Tx and i == Ty:
                    print("| T ", end="")
                    flag_T = 1
                
            for B in ssl:
                Bx = B[0]
                By = B[1]
                if j == Bx and i == By:
                    print("| B ", end="")
                    flag_B = 1
            
            for F in psl:
                Fx = F[0]
                Fy = F[1]
                if j == Fx and i == Fy:
                    print("| F ", end="")
                    flag_F = 1

            if j == spx and i == spy:
                print("| S ", end="")
            else:
                if flag_T == 0 and flag_F == 0 and flag_B == 0:
                    print("|   ", end="")

        print("|")
        
        for j in range(x):
            print("+---", end="")
        print("+")


draw_map(world_size_row, world_size_column, 
         start_point_row, start_point_column,
         terminal_state_list,
         prohibited_state_list,
         special_state_list)