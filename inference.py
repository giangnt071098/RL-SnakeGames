import random
from Snake import SnakeGame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
random.seed(10)
def main():
    # Animate games at different episodes
    with open("Qtable.npy", 'rb') as f:
        Qs = np.load(f)
    print("Generating data for animation...")
    boardDim = 16
    plotEpisodes = 10000
    fig, axes = plt.subplots(1, 1, figsize=(20, 20))

    dataArrays = []
    scores = []

    # Draw
    axes.set_title("Episode " + str(plotEpisodes), fontsize=30)
    axes.get_yaxis().set_visible(False)
    axes.get_xaxis().set_visible(False)

    ims = axes.imshow(np.zeros([boardDim, boardDim]), vmin=-1, vmax=1, cmap='RdGy')
    labels = axes.text(0, 15, "Length: 0", bbox={'facecolor': 'w', 'alpha': 0.75, 'pad': 1, 'edgecolor': 'white'}, fontsize=30)


    stopAnimation = False
    maxFrames = 1000
    cutoff = 100
    numGames = 1

    def animate(frameNum):
        labels.set_text("Length: " + str(scores[frameNum]))
        ims.set_data(dataArrays[frameNum])
        return [ims] + [labels]
    for k in range(numGames):
        games = SnakeGame(boardDim, boardDim)
        states = games.calcStateNum()
        # gameOver = False
        moveCounters = 0
        oldScores = 0

        for j in range(maxFrames):
            possibleQs = Qs[plotEpisodes, :, :][states, :]
            action = np.argmax(possibleQs)
            states, reward, gameOver, score = games.makeMove(action)
            dataArrays.append(games.plottableBoard())
            scores.append(score)
            if score == oldScores:
                moveCounters += 1
            else:
                oldScores = score
                moveCounters = 0
            if moveCounters >= cutoff:
                # stuck going back and forth
                gameOver = True
            if gameOver == True:
                print(f"Game{k}, finished, total moves: {len(dataArrays)}, score {score} " )
                break

    print("Animating snakes at different episodes...")

    numFrames = len(dataArrays)
    ani = animation.FuncAnimation(fig, func=animate, frames=numFrames, blit=True, interval=75, repeat=False, )
    plt.show(block=False)

    ##uncomment below if you want to output to a video file
    # print("Saving to file")
    # ani.save('AnimatedGames.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
    # print("Done")
if __name__ == '__main__':
    main()