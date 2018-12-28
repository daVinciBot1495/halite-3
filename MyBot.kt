import hlt.*
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Random
import kotlin.collections.ArrayList

enum class HaliteLevel constructor(val charValue: Char) {
    LOW('l'),
    MED('m'),
    HIGH('h');

    companion object {
        fun fromHalite(halite: Int): HaliteLevel {
            val validHalite = if (halite >= 0) halite else throw IllegalArgumentException("halite must be >= 0")

            return if (validHalite < 333) {
                HaliteLevel.LOW
            } else if (validHalite < 666) {
                HaliteLevel.MED
            } else {
                HaliteLevel.HIGH
            }
        }
    }
}

data class State(
        val maxHaliteDirection: Direction,
        val nearestDropOffDirection: Direction,
        val haliteLevel: HaliteLevel)

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

data class StateAction(val state: State, val action: Action)

interface ActionValueFunction {
    fun max(state: State, possibleActions: Set<Action>): Action

    fun update(state: State, action: Action, reward: Double, nextState: State, possibleNextActions: Set<Action>)

    fun saveState(path: Path)

    fun loadState(path: Path)
}

class TableActionValueFunction(
        private val discountRate: Double,
        private val learningRate: Double,
        private val stateActionValueMap: MutableMap<StateAction, Double> = mutableMapOf()) :
        ActionValueFunction {
    override fun max(state: State, possibleActions: Set<Action>): Action {
        if (possibleActions.isNotEmpty()) {
            val actionValues = possibleActions.map { Pair(it, getValue(state, it)) }
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
        val stateActionVal = getValue(state, action)
        val nextAction = max(nextState, possibleNextActions)
        val nextStateActionVal = getValue(nextState, nextAction)
        val temporalDiff = reward + discountRate * nextStateActionVal - stateActionVal

        setValue(state, action, stateActionVal + learningRate * temporalDiff)
    }

    override fun saveState(path: Path) {
        val text = stateActionValueMap.map { (stateAction, value) ->
            val keyStr = stateActionToStr(stateAction)
            val valueStr = value.toString()
            "$keyStr:$valueStr"
        }.joinToString(separator = "\n")
        path.toFile().writeText(text)
    }

    override fun loadState(path: Path) {
        path.toFile().forEachLine { line ->
            val trimmedLine = line.trim()

            if (trimmedLine.isNotEmpty()) {
                val tokens = trimmedLine.split(":")
                val keyStr = tokens[0]
                val valueStr = tokens[1]
                val stateActionPair = toStateAction(keyStr)
                val value = valueStr.toDouble()

                stateActionValueMap[stateActionPair] = value
            }
        }
    }

    private fun getValue(state: State, action: Action): Double {
        val stateActionPair = StateAction(state, action)
        return stateActionValueMap.getOrDefault(stateActionPair, 0.0)
    }

    private fun setValue(state: State, action: Action, value: Double) {
        val stateActionPair = StateAction(state, action)
        stateActionValueMap[stateActionPair] = value
    }

    private fun stateActionToStr(stateAction: StateAction): String {
        val state = stateAction.state
        val maxHaliteDirectionStr = state.maxHaliteDirection.charValue
        val nearestDropOffDirectionStr = state.nearestDropOffDirection.charValue
        val haliteLevelStr = state.haliteLevel.charValue
        val action = stateAction.action
        val actionStr = when (action) {
            is Move -> action.direction.charValue
        }

        return "$maxHaliteDirectionStr$nearestDropOffDirectionStr$haliteLevelStr$actionStr"
    }

    private fun toStateAction(str: String): StateAction {
        val state = State(toDirection(str[0]), toDirection(str[1]), toHaliteLevel(str[2]))
        val action = Move(toDirection(str[3]))
        return StateAction(state, action)
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

    private fun toHaliteLevel(c: Char): HaliteLevel {
        return when (c) {
            HaliteLevel.LOW.charValue -> HaliteLevel.LOW
            HaliteLevel.MED.charValue -> HaliteLevel.MED
            HaliteLevel.HIGH.charValue -> HaliteLevel.HIGH
            else -> throw IllegalArgumentException("$c is not a valid HaliteLevel")
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

fun toState(ship: Ship, shipyard: Shipyard, gameMap: GameMap): State {
    val directionHalitePairs = Direction.ALL_CARDINALS.map {
        val position = ship.position.directionalOffset(it)
        val mapCell = gameMap.at(position)
        val halite = (mapCell?.halite?.toDouble() ?: 0.0)
        Pair(it, halite)
    }
    val maxHaliteDirection = directionHalitePairs.maxBy { it.second }?.first!!
    val shipyardDirections = gameMap.getUnsafeMoves(ship.position, shipyard.position)
    val nearestDropOffDirection = if (shipyardDirections.isEmpty()) Direction.STILL else shipyardDirections.first()

    return State(maxHaliteDirection, nearestDropOffDirection, HaliteLevel.fromHalite(ship.halite))
}

fun toCommand(ship: Ship, action: Action): Command {
    return when (action) {
        is Move -> ship.move(action.direction)
    }
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
        val actionValueFunction = TableActionValueFunction(0.9,0.001)

        if (Files.exists(agentStatePath)) {
            Log.log("Loading agent state from ${agentStatePath.fileName}")
            actionValueFunction.loadState(agentStatePath)
        }

        val agent = QLearner(random, 0.30, actionValueFunction)
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

            me.shipyard

            //----------------------------------------------------------------------------------------------------------
            // Start Reinforcement Learning
            //----------------------------------------------------------------------------------------------------------
            val ship = me.ships.values.firstOrNull()

            if (ship == null) {
                commandQueue.add(me.shipyard.spawn())
                game.endTurn(commandQueue)
                continue
            }

            val currHalite = me.halite
            val currState = toState(ship, me.shipyard, gameMap)
            val currPossibleActions = toPossibleActions(ship, gameMap)
            val currAction = agent.chooseAction(currState, currPossibleActions)

            Log.log("Taking action=$currAction")

            if (prevState != null) {
                val reward = currHalite - prevHalite - 1
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
