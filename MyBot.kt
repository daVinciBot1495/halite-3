import hlt.*
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Random
import kotlin.collections.ArrayList

const val RADIUS = 5

enum class HaliteLevel constructor(val charValue: Char) {
    LOW('l'),
    MED('m'),
    HIGH('h');

    companion object {
        fun fromHalite(halite: Int): HaliteLevel {
            val validHalite = if (halite >= 0) halite else throw IllegalArgumentException("halite must be >= 0")

            return if (validHalite < 200) {
                LOW
            } else if (validHalite < 400) {
                MED
            } else {
                HIGH
            }
        }
    }
}

enum class DropOffProximity constructor(val charValue: Char) {
    CLOSE('c'),
    FAR('f');

    companion object {
        fun fromDistance(distance: Int): DropOffProximity {
            return if (distance <= 5) {
                CLOSE
            } else {
                FAR
            }
        }
    }
}

data class State(
        val directionHaliteLevels: List<HaliteLevel>,
        val nearestDropOffDirection: Direction,
        val nearestDropOffProximity: DropOffProximity,
        val shipHaliteLevel: HaliteLevel)

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

    fun update(
            state: State,
            action: Action,
            reward: Double,
            nextState: State,
            nextAction: Action,
            possibleNextActions: Set<Action>)

    fun saveState(path: Path)

    fun loadState(path: Path)
}

class TableActionValueFunction(
        private val discountRate: Double,
        private val learningRate: Double,
        private val traceDecayRate: Double,
        private val stateActionValueMap: MutableMap<StateAction, Double> = mutableMapOf(),
        private val stateActionTraceMap: MutableMap<StateAction, Double> = mutableMapOf(),
        private val episode: MutableSet<StateAction> = mutableSetOf()) :
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
            nextAction: Action,
            possibleNextActions: Set<Action>) {
        val stateActionVal = getValue(state, action)
        val nextOptimalAction = max(nextState, possibleNextActions)
        val nextOptimalStateActionVal = getValue(nextState, nextOptimalAction)
        val temporalDiff = reward + discountRate * nextOptimalStateActionVal - stateActionVal

        Log.log("s=$state, a=$action, r=$reward, s'=$nextState, a'=$nextAction, a*=$nextOptimalAction")

        setTrace(state, action, getTrace(state, action) + 1.0)

        episode.add(StateAction(state, action))
        episode.forEach {
            val oldVal = getValue(it.state, it.action)
            val oldTrace = getTrace(it.state, it.action)
            val newVal = oldVal + learningRate * temporalDiff * oldTrace
            val newTrace = if (nextAction == nextOptimalAction) {
                discountRate * traceDecayRate * oldTrace
            } else {
                0.0
            }

            Log.log("Updating: s=${it.state}, a=${it.action}, oldVal=$oldVal, newVal=$newVal")

            setValue(it.state, it.action, newVal)
            setTrace(it.state, it.action, newTrace)
        }

        if (reward >= 0) {
            episode.forEach { setTrace(it.state, it.action, 0.0) }
            episode.clear()
        }
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

    private fun getTrace(state: State, action: Action): Double {
        val stateActionPair = StateAction(state, action)
        return stateActionTraceMap.getOrDefault(stateActionPair, 0.0)
    }

    private fun setTrace(state: State, action: Action, value: Double) {
        val stateActionPair = StateAction(state, action)
        stateActionTraceMap[stateActionPair] = value
    }

    private fun stateActionToStr(stateAction: StateAction): String {
        val state = stateAction.state
        val directionHaliteLevelsStr = directionHaliteLevelsToStr(state.directionHaliteLevels)
        val nearestDropOffDirectionStr = state.nearestDropOffDirection.charValue
        val nearestDropOffProximityStr = state.nearestDropOffProximity.charValue
        val shipHaliteLevelStr = state.shipHaliteLevel.charValue
        val action = stateAction.action
        val actionStr = when (action) {
            is Move -> action.direction.charValue
        }

        return "$directionHaliteLevelsStr$nearestDropOffDirectionStr$nearestDropOffProximityStr$shipHaliteLevelStr$actionStr"
    }

    private fun directionHaliteLevelsToStr(directionHaliteLevels: List<HaliteLevel>): String {
        return directionHaliteLevels.map { it.charValue }.joinToString(separator = "")
    }

    private fun toStateAction(str: String): StateAction {
        val directionHaliteLevels = listOf(
                toHaliteLevel(str[0]),
                toHaliteLevel(str[1]),
                toHaliteLevel(str[2]),
                toHaliteLevel(str[3]),
                toHaliteLevel(str[4]))
        val nearestDropOffDirection = toDirection(str[5])
        val nearestDropOffProximity = toDropOffProximity(str[6])
        val shipHaliteLevel = toHaliteLevel(str[7])
        val state = State(directionHaliteLevels, nearestDropOffDirection, nearestDropOffProximity, shipHaliteLevel)
        val action = Move(toDirection(str[8]))
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

    private fun toDropOffProximity(c: Char): DropOffProximity {
        return when (c) {
            DropOffProximity.CLOSE.charValue -> DropOffProximity.CLOSE
            DropOffProximity.FAR.charValue -> DropOffProximity.FAR
            else -> throw IllegalArgumentException("$c is not a valid DropOffProximity")
        }
    }
}

interface ReinforcementLearner {
    fun chooseAction(currState: State, possibleNextActions: Set<Action>): Action

    fun learn(
            currState: State,
            action: Action,
            reward: Double,
            nextState: State,
            nextAction: Action,
            possibleNextActions: Set<Action>)

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
            nextAction: Action,
            possibleNextActions: Set<Action>) {
        actionValueFunction.update(currState, action, reward, nextState, nextAction, possibleNextActions)
    }

    override fun saveState(path: Path) {
        actionValueFunction.saveState(path)
    }

    override fun loadState(path: Path) {
        actionValueFunction.loadState(path)
    }
}

fun Position.directionalOffset(d: Direction, scale: Int): Position {
    val dx: Int
    val dy: Int

    when (d) {
        Direction.NORTH -> {
            dx = 0
            dy = -scale
        }
        Direction.SOUTH -> {
            dx = 0
            dy = scale
        }
        Direction.EAST -> {
            dx = scale
            dy = 0
        }
        Direction.WEST -> {
            dx = -scale
            dy = 0
        }
        Direction.STILL -> {
            dx = 0
            dy = 0
        }
    }

    return Position(x + dx, y + dy)
}

fun toState(ship: Ship, shipyard: Shipyard, gameMap: GameMap): State {
    val directions = listOf(Direction.STILL, Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST)
    val directionHaliteLevels = directions.map {
        var num = 0.0
        var denom = 0.0

        if (it == Direction.STILL) {
            val mapCell = gameMap.at(ship)
            num += mapCell?.halite?.toDouble() ?: 0.0
            denom += 1.0
        } else {
            for (scale in 1..RADIUS) {
                val position = ship.position.directionalOffset(it, scale)
                val mapCell = gameMap.at(position)
                val weight = RADIUS - scale + 1.0
                num += weight * (mapCell?.halite?.toDouble() ?: 0.0)
                denom += weight
            }
        }

        val directionHalite = (num / denom).toInt()
        val directionHaliteLevel = HaliteLevel.fromHalite(directionHalite)

        directionHaliteLevel
    }
    val shipyardDirections = gameMap.getUnsafeMoves(ship.position, shipyard.position)
    val nearestDropOffDirection = if (shipyardDirections.isEmpty()) Direction.STILL else shipyardDirections.first()
    val nearestDropOffDistance = gameMap.calculateDistance(ship.position, shipyard.position)
    val nearestDropOffProximity = DropOffProximity.fromDistance(nearestDropOffDistance)
    val shipHaliteLevel = HaliteLevel.fromHalite(ship.halite)

    return State(directionHaliteLevels, nearestDropOffDirection, nearestDropOffProximity, shipHaliteLevel)
}

fun toCommand(ship: Ship, action: Action): Command {
    return when (action) {
        is Move -> ship.move(action.direction)
    }
}

fun toPossibleActions(ship: Ship, shipyard: Shipyard, gameMap: GameMap): Set<Action> {
    val mapCell = gameMap.at(ship.position)
    val cellHalite = mapCell?.halite?.toDouble() ?: 0.0

    return if (ship.halite < cellHalite / 10) {
        setOf(Action.STILL)
    } else if (ship.position == shipyard.position) {
        setOf(Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST)
    } else {
        Action.ALL_ACTIONS
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
        val actionValueFunction = TableActionValueFunction(0.9,0.01, 0.9)

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
            val currPossibleActions = toPossibleActions(ship, me.shipyard, gameMap)
            val currAction = agent.chooseAction(currState, currPossibleActions)

            if (prevState != null) {
                val reward = currHalite - prevHalite - 1
                agent.learn(prevState, prevAction, reward.toDouble(), currState, currAction, currPossibleActions)
            }

            prevState = currState
            prevAction = currAction
            prevHalite = currHalite

            commandQueue.add(toCommand(ship, currAction))

            if (saveState && Constants.MAX_TURNS == game.turnNumber) {
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
