import hlt.*
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Random
import kotlin.collections.ArrayList

val twoByTwoTiles = listOf(
        Tile("2.2.NW", TileRange(-1, 0), TileRange(-1,0)),
        Tile("2.2.NE", TileRange(0, 1), TileRange(-1, 0)),
        Tile("2.2.SE", TileRange(0, 1), TileRange(0, 1)),
        Tile("2.2.SW", TileRange(-1, 0), TileRange(0, 1)))

data class TileRange(val start: Int, val end: Int) {
    init {
        if (end < start) {
            throw IllegalArgumentException("start must be <= end")
        }
    }
}

data class Tile(val id: String, val xRange: TileRange, val yRange: TileRange) {
    init {
        if (id.isBlank()) {
            throw IllegalArgumentException("id cannot be blank")
        }
    }

    fun getPositionsForAnchor(anchor: Position): List<Position> {
        val positions = mutableListOf<Position>()

        for (i in xRange.start..xRange.end) {
            for (j in yRange.start..yRange.end) {
                positions.add(add(anchor, i, j))
            }
        }

        return positions
    }

    private fun add(position: Position, x: Int, y: Int): Position {
        return Position(position.x + x, position.y + y)
    }
}

data class RankedTile(val rank: Int, val tile: Tile) {
    init {
        if (rank < 0) {
            throw IllegalArgumentException("rank must be >= 0")
        }
    }
}

data class HaliteLevel(val rank: Int) {
    init {
        if (rank < 0 || rank > THRESHOLDS.size) {
            throw IllegalArgumentException("rank=$rank must be >= 0 and <= ${THRESHOLDS.size}")
        }
    }

    companion object {
        private val THRESHOLDS = listOf(8, 16, 32, 64, 128, 256, 512)

        fun fromHalite(halite: Int): HaliteLevel {
            val index = THRESHOLDS.binarySearch(halite)

            return if (index >= 0) {
                HaliteLevel(index)
            } else {
                val insertionPoint = -(index + 1)
                HaliteLevel(insertionPoint)
            }
        }
    }
}

data class State(
        val twoByTwoRankedTiles: List<RankedTile>,
        val nearestDropOffDirection: Direction,
        val shipHaliteLevel: HaliteLevel,
        val cellHaliteLevel: HaliteLevel)

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

        if (reward != 0.0) {
            Log.log("Clearing episode due to non-zero reward=$reward")
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
        val twoByTwoRankedTilesStr = rankedTilesToStr(state.twoByTwoRankedTiles)
        val nearestDropOffDirectionStr = state.nearestDropOffDirection.charValue
        val shipHaliteLevelStr = state.shipHaliteLevel.rank.toString()
        val cellHaliteLevelStr = state.cellHaliteLevel.rank.toString()
        val action = stateAction.action
        val actionStr = when (action) {
            is Move -> action.direction.charValue
        }

        return "$twoByTwoRankedTilesStr$nearestDropOffDirectionStr$shipHaliteLevelStr$cellHaliteLevelStr$actionStr"
    }

    private fun rankedTilesToStr(rankedTiles: List<RankedTile>): String {
        return rankedTiles.map { it.rank }.joinToString(separator = "")
    }

    private fun toStateAction(str: String): StateAction {
        var index = 0
        val twoByTwoRankedTiles = twoByTwoTiles.map { RankedTile(str[index++].toString().toInt(), it) }
        val nearestDropOffDirection = toDirection(str[index++])
        val shipHaliteLevel = HaliteLevel(str[index++].toString().toInt())
        val cellHaliteLevel = HaliteLevel(str[index++].toString().toInt())
        val state = State(twoByTwoRankedTiles, nearestDropOffDirection, shipHaliteLevel, cellHaliteLevel)
        val action = Move(toDirection(str[index++]))
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

fun toState(ship: Ship, shipyard: Shipyard, gameMap: GameMap): State {
    val twoByTwoTileHaliteAvgs = twoByTwoTiles.map { tile ->
        val positions = tile.getPositionsForAnchor(ship.position)
        val haliteSum = positions.sumBy { gameMap.at(it)?.halite ?: 0 }
        Pair(tile, haliteSum.toDouble() / positions.size)
    }.toMutableList()
    twoByTwoTileHaliteAvgs.sortBy { it.second }
    val tileToRankMap = twoByTwoTileHaliteAvgs.mapIndexed { index, pair ->  pair.first to index }.toMap()
    val twoByTwoRankedTiles = twoByTwoTiles.map { RankedTile(tileToRankMap[it]!!, it) }

    val shipyardDirections = gameMap.getUnsafeMoves(ship.position, shipyard.position)
    val nearestDropOffDirection = if (shipyardDirections.isEmpty()) Direction.STILL else shipyardDirections.first()

    val shipHaliteLevel = HaliteLevel.fromHalite(ship.halite)

    val cellHaliteLevel = HaliteLevel.fromHalite(gameMap.at(ship)?.halite ?: 0)

    return State(twoByTwoRankedTiles, nearestDropOffDirection, shipHaliteLevel, cellHaliteLevel)
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

        val agent = QLearner(random, 0.01, actionValueFunction)
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
                val reward = currHalite - prevHalite
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
