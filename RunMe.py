import gym
import gym_gvgai
import time
import sso as sso_class
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from A3CAgent.A3CAgent import A3CAgent
#plt.style.use('dark_background')

games = [
    'gvgai-labyrinthdual', 'gvgai-labyrinth',
    'gvgai-lasers2', 'gvgai-lasers', 'gvgai-lemmings', 'gvgai-mirrors', 'gvgai-missilecommand', 'gvgai-modality',
    'gvgai-overload', 'gvgai-pacman', 'gvgai-painter', 'gvgai-plants', 'gvgai-plaqueattack', 'gvgai-pokemon', 'gvgai-portals',
    'gvgai-racebet2', 'gvgai-racebet', 'gvgai-realportals', 'gvgai-realsokoban', 'gvgai-rivers', 'gvgai-roadfighter', 'gvgai-roguelike',
    'gvgai-run', 'gvgai-seaquest', 'gvgai-sheriff', 'gvgai-shipwreck', 'gvgai-sistersavior', 'gvgai-sokoban', 'gvgai-solarfox',
    'gvgai-superman', 'gvgai-surround', 'gvgai-survivezombies', 'gvgai-tercio', 'gvgai-testgame1', 'gvgai-testgame2', 'gvgai-testgame3',
    'gvgai-thecitadel', 'gvgai-themole', 'gvgai-theshepherd', 'gvgai-thesnowman', 'gvgai-vortex', 'gvgai-waferthinmintsexit',
    'gvgai-waferthinmints', 'gvgai-waitforbreakfast', 'gvgai-watergame', 'gvgai-waves', 'gvgai-whackamole', 'gvgai-wildgunman',
    'gvgai-witnessprotected', 'gvgai-witnessprotection', 'gvgai-wrapsokoban', 'gvgai-x-racer', 'gvgai-zelda', 'gvgai-zenpuzzle'] # add 10 games
#performs good on : "gvgai-aliens","gvgai-boulderchase","gvgai-brainman",
#   "gvgai-defem","gvgai-donkeykong",'gvgai-ikaruga','gvgai-invest',
#do not add(crash) : 'gvgai-ghostbuster','gvgai-killBillVol1',



results_file = open("results.txt", "w")
results_file.write("") # empty file
results_file.close()

results_file = open("results.txt", "a")

fig, ax = plt.subplots()
xs = [0]
ys = [0]

ax.set_xlabel("Game Ticks")
ax.set_ylabel("Score")
def animate(i):
    line, = ax.plot(xs, ys)
    return line,

ani = animation.FuncAnimation(fig, animate, interval=2)
plt.show(block=False)

for game in games:
    ax.clear()
    print("Playing game "+game)
    results_file.write("Playing game "+game+"\n")
    #run level 0,1,2 for training for 5 minutes
    training_time_limit = 5*60
    training_start_time = time.time()
    elapsed_timer = 0
    Agent = A3CAgent()
    next_level = 0
    #run levels 0,1,2 in sequence
    for i in range(3):
        print("Training : Playing "+game+" level "+str(i))
        ax.set_title("Training : Game :"+game+" Level :"+str(i))
        results_file.write("Training : level "+str(i)+" Score : ")

        env = gym.make(game + "-lvl" + str(i) + "-v0")
        observation = env.reset()

        sso = sso_class.sso()
        sso.availableActions = [x for x in range(env.action_space.n)]#just send a number
        sso.observation = observation
        Agent.init(sso, elapsed_timer)
        xs = [0]
        ys = [0]
        
        for j in range (1000):
            
            print("tick "+str(sso.gameTick))
            #env.render()

            sso.gameTick += 1
            # print("Training : Running "+game+" lvl :"+str(i)+" Tick :"+str(sso.gameTick)+" Score :"+str(sso.gameScore))
            observation, reward, done, info = env.step(Agent.act(sso, elapsed_timer))
            #reward is -1.0 when player loses
            #observation is visual screen
            sso.gameScore += reward
            sso.observation = observation
            
            xs.append(sso.gameTick)
            ys.append(sso.gameScore)

            elapsed_timer = time.time() - training_start_time

            if j == 999:
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                results_file.write(str(sso.gameScore)+"\n")
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                #close render window
                env.close()
                break
            if done:
                if reward != -1:
                    sso.gameWinner = "Player"
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                results_file.write(str(sso.gameScore)+"\n")
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                #close render window
                env.close()
                break
            if elapsed_timer > training_time_limit:
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                #close render window
                env.close()
                break


        elapsed_timer = time.time() - training_start_time
        if elapsed_timer > training_time_limit:
            break

    elapsed_timer = time.time() - training_start_time
    while elapsed_timer < training_time_limit and False:
        if next_level > 2 and next_level < 0:
            next_level = 0
        print("Training : Playing "+game+" level "+str(next_level))
        ax.set_title("Training : Game :"+game+" Level :"+str(next_level))
        results_file.write("Training : level "+str(next_level)+" Score : ")

        env = gym.make(game + "-lvl" + str(next_level) + "-v0")
        observation = env.reset()

        sso = sso_class.sso()
        sso.availableActions = [x for x in range(env.action_space.n)]#just send a number
        sso.observation = observation
        Agent.init(sso, elapsed_timer)
        xs = [0]
        ys = [0]

        for j in range (1000):
            
            print("tick "+str(sso.gameTick))
            # env.render()

            sso.gameTick += 1
            #print("Training : Running "+game+" lvl :"+str(next_level)+" Tick :"+str(sso.gameTick)+" Score :"+str(sso.gameScore))
            observation, reward, done, info = env.step(Agent.act(sso, elapsed_timer))
            #reward is -1.0 when player loses
            #observation is visual screen
            sso.gameScore += reward
            sso.observation = observation
            
            xs.append(sso.gameTick)
            ys.append(sso.gameScore)

            elapsed_timer = time.time() - training_start_time

            if j == 999:
                print("Training : level "+str(next_level)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                # env.close()
                break
            if done:
                if reward != -1:
                    sso.gameWinner = "Player"
                print("Training : level "+str(next_level)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                elapsed_timer = time.time() - training_start_time
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                # env.close()
                break
            if elapsed_timer > training_time_limit:
                print("Training : level "+str(next_level)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                # env.close()
                break

    #run level 3,4 for validation
    evaluvating_time_limit = 2*60
    evaluvating_start_time = time.time()
    elapsed_timer = 0
    for i in range(3,5):
        print("Evaluvation : Playing "+game+" level "+str(i))
        ax.set_title("Evaluation : Game :"+game+" Level :"+str(i))
        results_file.write("Evaluation : level "+str(i)+" Score : ")

        env = gym.make(game + "-lvl" + str(i) + "-v0")
        observation = env.reset()

        sso = sso_class.sso()
        sso.availableActions = [x for x in range(env.action_space.n)]#just send a number
        sso.observation = observation
        Agent.init(sso, elapsed_timer)
        xs = [0]
        ys = [0]

        for j in range (1000):
            
            print("tick "+str(sso.gameTick))
            # env.render()

            sso.gameTick += 1
            #print("Evaluvation : Running "+game+" lvl :"+str(i)+" Tick :"+str(sso.gameTick)+" Score :"+str(sso.gameScore))
            observation, reward, done, info = env.step(Agent.act(sso, elapsed_timer))
            #reward is -1.0 when player loses
            #observation is visual screen
            sso.gameScore += reward
            sso.observation = observation
            
            xs.append(sso.gameTick)
            ys.append(sso.gameScore)

            elapsed_timer = time.time() - evaluvating_start_time

            if j == 999:
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                env.close()
                break
            if done:
                if reward != -1:
                    sso.gameWinner = "Player"
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                # env.close()
                break
            if elapsed_timer > evaluvating_time_limit:
                print("Training : level "+str(i)+" Score : "+str(sso.gameScore))
                next_level = Agent.result(sso, elapsed_timer)
                results_file.write(str(sso.gameScore)+"\n")
                #close render window
                # env.close()
                break

    
results_file.close()