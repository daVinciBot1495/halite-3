import hlt.*
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Random
import kotlin.collections.ArrayList

data class Vector(val elements: List<Double>) {
    fun scale(s: Double): Vector {
        return Vector(this.elements.map { s * it })
    }

    fun add(v: Vector): Vector {
        if (this.size() != v.size()) {
            throw IllegalArgumentException("The vector sizes must be equal: ${this.size()} != ${v.size()}")
        }
        return Vector(this.elements.zip(v.elements).map { it.first + it.second })
    }

    fun dot(v: Vector): Double {
        if (this.size() != v.size()) {
            throw IllegalArgumentException("The vector sizes must be equal: ${this.size()} != ${v.size()}")
        }
        return this.elements.zip(v.elements).map { it.first * it.second }.reduce { a, b -> a + b }
    }

    companion object {
        fun random(n: Int, r: Random): Vector {
            return Vector(DoubleArray(n, { r.nextDouble() }).toList())
        }
    }

    private fun size(): Int {
        return this.elements.size
    }
}

data class State(val features: Vector)

sealed class Action {
    companion object {
        val STILL: Action = Move(Direction.STILL)
        val NORTH: Action = Move(Direction.NORTH)
        val SOUTH: Action = Move(Direction.SOUTH)
        val EAST: Action = Move(Direction.EAST)
        val WEST: Action = Move(Direction.WEST)
        val ALL_ACTIONS = setOf(STILL, NORTH, SOUTH, EAST, WEST)
    }
}

data class Move(val direction: Direction) : Action()

interface ActionValueFunction {
    fun evaluate(state: State, action: Action): Double

    fun max(state: State, possibleActions: Set<Action>): Action

    fun update(state: State, action: Action, reward: Double, nextState: State, possibleNextActions: Set<Action>)

    fun saveState(path: Path)

    fun loadState(path: Path)
}

class LinearActionValueFunction(
        private val random: Random,
        private val discountRate: Double,
        private val learningRate: Double,
        private val actionParamsMap: MutableMap<Action, Vector> = mutableMapOf()) :
        ActionValueFunction {
    override fun evaluate(state: State, action: Action): Double {
        return getParams(state, action).dot(state.features)
    }

    override fun max(state: State, possibleActions: Set<Action>): Action {
        if (possibleActions.isNotEmpty()) {
            val actionValues = possibleActions.map { Pair(it, evaluate(state, it)) }
            return actionValues.maxBy { it.second }?.first!!
        } else {
            throw IllegalArgumentException("possibleActions cannot be empty")
        }
    }

    override fun update(
            state: State,
            action: Action,
            reward: Double,
            nextState: State,
            possibleNextActions: Set<Action>) {
        val params = getParams(state, action)
        val actionVal = evaluate(state, action)
        val nextAction = max(nextState, possibleNextActions)
        val nextActionVal = evaluate(nextState, nextAction)
        val temporalDiff = reward + discountRate * nextActionVal - actionVal
        val newParams = params.add(state.features.scale(learningRate * temporalDiff))

        actionParamsMap[action] = newParams
    }

    override fun saveState(path: Path) {
        val text = actionParamsMap.map { (action, vector) ->
            val valuesStr = vector.elements.map { it.toString() }.joinToString(",")
            val keyStr = when (action) {
                is Move -> action.direction.charValue
            }
            "$keyStr:$valuesStr"
        }.joinToString(separator = "\n")
        path.toFile().writeText(text)
    }

    override fun loadState(path: Path) {
        path.toFile().forEachLine { line ->
            val trimmedLine = line.trim()

            if (trimmedLine.isNotEmpty()) {
                val tokens = trimmedLine.split(":")
                val keyStr = tokens[0]
                val valuesStr = tokens[1]
                val action = Move(toDirection(keyStr[0]))
                val values = Vector(valuesStr.split(",").map { valStr -> valStr.toDouble() })

                actionParamsMap[action] = values
            }
        }
    }

    private fun getParams(state: State, action: Action): Vector {
        return actionParamsMap.computeIfAbsent(action, { _ ->
            Vector.random(state.features.elements.size, random)
        })
    }

    private fun toDirection(c: Char): Direction {
        return when (c) {
            Direction.STILL.charValue -> Direction.STILL
            Direction.NORTH.charValue -> Direction.NORTH
            Direction.SOUTH.charValue -> Direction.SOUTH
            Direction.EAST.charValue -> Direction.EAST
            Direction.WEST.charValue -> Direction.WEST
            else -> throw IllegalArgumentException("$c is not a valid Direction")
        }
    }
}

interface ReinforcementLearner {
    fun chooseAction(currState: State, possibleNextActions: Set<Action>): Action

    fun learn(currState: State, action: Action, reward: Double, nextState: State, possibleNextActions: Set<Action>)

    fun saveState(path: Path)

    fun loadState(path: Path)
}

class QLearner(
        private val random: Random,
        private val explorationRate: Double,
        private val actionValueFunction: ActionValueFunction) : ReinforcementLearner {
    override fun chooseAction(currState: State, possibleNextActions: Set<Action>): Action {
        if (possibleNextActions.isEmpty()) {
            throw IllegalArgumentException("possibleNextActions cannot be empty")
        }

        return if (random.nextDouble() < explorationRate) {
            Log.log("Taking exploratory action")
            Action.ALL_ACTIONS.toList()[random.nextInt(Action.ALL_ACTIONS.size)]
        } else {
            actionValueFunction.max(currState, possibleNextActions)
        }
    }

    override fun learn(
            currState: State,
            action: Action,
            reward: Double,
            nextState: State,
            possibleNextActions: Set<Action>) {
        actionValueFunction.update(currState, action, reward, nextState, possibleNextActions)
    }

    override fun saveState(path: Path) {
        actionValueFunction.saveState(path)
    }

    override fun loadState(path: Path) {
        actionValueFunction.loadState(path)
    }
}

fun toState(ship: Ship, gameMap: GameMap): State {
    val features = mutableListOf<Double>()
    val mask = listOf(-4, -3, -2, -1, 0, 1, 2, 3, 4)

    for (i in mask) {
        for (j in mask) {
            val position = ship.position.add(i, j)
            val mapCell = gameMap.at(position)
            val halite = (mapCell?.halite?.toDouble() ?: 0.0) / Constants.MAX_HALITE
            val isOccupied = mapCell?.isOccupied ?: false
            val isDepositLoc = mapCell?.hasStructure() ?: false

            features.add(halite)
            features.add(if (isOccupied) 1.0 else 0.0)
            features.add(if (isDepositLoc) 1.0 else 0.0)
        }
    }

    features.add(ship.halite.toDouble() / Constants.MAX_HALITE)
    features.add(1.0)

    return State(Vector(features))
}

fun toCommand(ship: Ship, action: Action): Command {
    return when (action) {
        is Move -> ship.move(action.direction)
    }
}

fun Position.add(x: Int, y: Int): Position {
    return Position(this.x + x, this.y + y)
}

fun toPossibleActions(ship: Ship, gameMap: GameMap): Set<Action> {
    val mapCell = gameMap.at(ship.position)
    val cellHalite = mapCell?.halite?.toDouble() ?: 0.0

    return if (ship.isFull) {
        setOf(Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST)
    } else if (ship.halite > cellHalite / 10) {
        Action.ALL_ACTIONS
    } else {
        setOf(Action.STILL)
    }
}

object MyBot {
    @JvmStatic
    fun main(args: Array<String>) {
        val rngSeed = System.nanoTime()
        val saveState: Boolean = if (args.isNotEmpty()) args[0].toBoolean() else true
        val random = Random(rngSeed)
        val game = Game()

        // At this point "game" variable is populated with initial map data.
        // This is a good place to do computationally expensive start-up pre-processing.
        // As soon as you call "ready" function below, the 2 second per turn timer will start.
        val agentStatePath = Paths.get("q.txt")
        val actionValueFunction = LinearActionValueFunction(
                random,
                0.999,
                0.000001)

        if (Files.exists(agentStatePath)) {
            Log.log("Loading agent state from ${agentStatePath.fileName}")
            actionValueFunction.loadState(agentStatePath)
        }

        val agent = QLearner(random, 0.05, actionValueFunction)
        var prevState: State? = null
        var prevAction = Action.STILL
        var prevHalite = 0

        game.ready("daVinciBot1495")

        Log.log("Successfully created bot! My Player ID is " + game.myId + ". Bot rng seed is " + rngSeed + ".")

        while (true) {
            game.updateFrame()

            val me = game.me
            val gameMap = game.gameMap
            val commandQueue = ArrayList<Command>()

            //----------------------------------------------------------------------------------------------------------
            // Start Reinforcement Learning
            //----------------------------------------------------------------------------------------------------------
            val ship = me.ships.values.firstOrNull()

            if (ship == null) {
                commandQueue.add(me.shipyard.spawn())
                game.endTurn(commandQueue)
                continue
            }

            val currHalite = me.halite + ship.halite
            val currState = toState(ship, gameMap)
            val currPossibleActions = toPossibleActions(ship, gameMap)
            val currAction = agent.chooseAction(currState, currPossibleActions)

            Log.log("Taking action=$currAction")

            if (prevState != null) {
                val reward = currHalite - prevHalite
                agent.learn(prevState, prevAction, reward.toDouble(), currState, currPossibleActions)
                Log.log("reward=$reward")
            }

            prevState = currState
            prevAction = currAction
            prevHalite = currHalite

            commandQueue.add(toCommand(ship, currAction))

            if (saveState) {
                Log.log("Saving state to ${agentStatePath.fileName}")
                agent.saveState(agentStatePath)
            } else {
                Log.log("Not saving state")
            }
            //----------------------------------------------------------------------------------------------------------
            // End Reinforcement Learning
            //----------------------------------------------------------------------------------------------------------

            game.endTurn(commandQueue)
        }
    }
}
