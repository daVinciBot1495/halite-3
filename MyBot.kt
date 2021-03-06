import hlt.*
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.Random
import kotlin.IllegalStateException
import kotlin.collections.ArrayList

val allDirections = setOf(Direction.STILL, Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST)

val twoByTwoTiles = listOf(
        Tile("2.2.NW", TileRange(-1, 0), TileRange(-1,0)),
        Tile("2.2.NE", TileRange(0, 1), TileRange(-1, 0)),
        Tile("2.2.SE", TileRange(0, 1), TileRange(0, 1)),
        Tile("2.2.SW", TileRange(-1, 0), TileRange(0, 1)))

data class AgentId(val id: Int) {
    init {
        if (id < 0) throw IllegalArgumentException("id must be >= 0")
    }
}

data class TileRange(val start: Int, val end: Int) {
    init {
        if (end < start) throw IllegalArgumentException("start must be <= end")
    }
}

data class Tile(val id: String, val xRange: TileRange, val yRange: TileRange) {
    init {
        if (id.isBlank()) throw IllegalArgumentException("id cannot be blank")
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
        if (rank < 0) throw IllegalArgumentException("rank must be >= 0")
    }
}

data class HaliteLevel(val rank: Int) {
    init {
        if (rank < 0 || rank > THRESHOLDS.size) throw IllegalArgumentException("rank=$rank must be >= 0 and <= ${THRESHOLDS.size}")
    }

    companion object {
        private val THRESHOLDS = listOf(16, 32, 64, 128, 256, 512)

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
        val agentId: AgentId,
        val twoByTwoRankedTiles: List<RankedTile>,
        val nearestDropOffDirection: Direction,
        val shipHaliteLevel: HaliteLevel,
        val cellHaliteLevel: HaliteLevel)

sealed class Action {
    companion object {
        val STILL: Action = Move(Direction.STILL)
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
        val terminal = reward != 0.0
        val stateActionVal = getValue(state, action)
        val nextStateActionVal = if (terminal) 0.0 else getValue(nextState, nextAction)
        val temporalDiff = reward + discountRate * nextStateActionVal - stateActionVal

        Log.log("s=$state, a=$action, r=$reward, s'=$nextState, a'=$nextAction")

        // Use a replacing trace
        setTrace(state, action, 1.0)

        episode.add(StateAction(state, action))
        episode.forEach {
            val oldVal = getValue(it.state, it.action)
            val oldTrace = getTrace(it.state, it.action)
            val newVal = oldVal + learningRate * temporalDiff * oldTrace
            val newTrace = discountRate * traceDecayRate * oldTrace

            Log.log("Updating: s=${it.state}, a=${it.action}, oldVal=$oldVal, newVal=$newVal")

            setValue(it.state, it.action, newVal)
            setTrace(it.state, it.action, newTrace)
        }

        if (terminal) {
            Log.log("Clearing episode since next state is a terminal state")
            episode.forEach { setTrace(it.state, it.action, 0.0) }
            episode.clear()
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
}

class ActionValueFunctionStore {
    fun save(path: Path, stateActionValueMap: MutableMap<StateAction, Double>) {
        path.toFile().bufferedWriter().use { out ->
            stateActionValueMap.forEach {
                val keyStr = stateActionToStr(it.key)
                val valueStr = it.value.toString()
                out.write("$keyStr:$valueStr")
                out.newLine()
            }

            out.flush()
        }
    }

    fun load(path: Path): MutableMap<StateAction, Double> {
        val stateActionValueMap = mutableMapOf<StateAction, Double>()

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

        return stateActionValueMap
    }

    private fun stateActionToStr(stateAction: StateAction): String {
        val state = stateAction.state
        val agentIdStr = state.agentId.id.toString()
        val twoByTwoRankedTilesStr = rankedTilesToStr(state.twoByTwoRankedTiles)
        val nearestDropOffDirectionStr = state.nearestDropOffDirection.charValue
        val shipHaliteLevelStr = state.shipHaliteLevel.rank.toString()
        val cellHaliteLevelStr = state.cellHaliteLevel.rank.toString()
        val action = stateAction.action
        val actionStr = when (action) {
            is Move -> action.direction.charValue
        }

        return "$agentIdStr$twoByTwoRankedTilesStr$nearestDropOffDirectionStr$shipHaliteLevelStr$cellHaliteLevelStr$actionStr"
    }

    private fun rankedTilesToStr(rankedTiles: List<RankedTile>): String {
        return rankedTiles.map { it.rank }.joinToString(separator = "")
    }

    private fun toStateAction(str: String): StateAction {
        var index = 0
        val agentId = AgentId(str[index++].toString().toInt())
        val twoByTwoRankedTiles = twoByTwoTiles.map { RankedTile(str[index++].toString().toInt(), it) }
        val nearestDropOffDirection = toDirection(str[index++])
        val shipHaliteLevel = HaliteLevel(str[index++].toString().toInt())
        val cellHaliteLevel = HaliteLevel(str[index++].toString().toInt())
        val state = State(agentId, twoByTwoRankedTiles, nearestDropOffDirection, shipHaliteLevel, cellHaliteLevel)
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
}

class SarsaLearner(
        private val random: Random,
        private val explorationRate: Double,
        private val actionValueFunction: ActionValueFunction) : ReinforcementLearner {
    override fun chooseAction(currState: State, possibleNextActions: Set<Action>): Action {
        if (possibleNextActions.isEmpty()) {
            throw IllegalArgumentException("possibleNextActions cannot be empty")
        }

        return if (random.nextDouble() < explorationRate) {
            possibleNextActions.toList()[random.nextInt(possibleNextActions.size)]
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
}

data class ContextId(val id: Int) {
    init {
        if (id < 0) throw IllegalArgumentException("id must be >= 0")
    }
}

class Context(val contextId: ContextId, val agent: ReinforcementLearner, var prevStateAction: StateAction?)

class ContextManager(
        private val random: Random,
        private val maxContextCount: Int,
        private val stateActionValueMap: MutableMap<StateAction, Double>,
        private val shipContextMap: MutableMap<EntityId, Context> = mutableMapOf()) {
    fun hasContextAvailable(): Boolean {
        return shipContextMap.size < maxContextCount
    }

    fun getShipIds(): Set<EntityId> {
        return shipContextMap.keys
    }

    fun newContext(shipId: EntityId): Context {
        val contextId = getNextContextId()
        val actionValueFunction = TableActionValueFunction(0.99, 0.01, 0.9, stateActionValueMap)
        val context = Context(contextId, SarsaLearner(random, 0.10, actionValueFunction),null)

        shipContextMap[shipId] = context

        Log.log("Created new context=${context.contextId} for shipId=$shipId")

        return context
    }

    fun getContext(shipId: EntityId): Context {
        return shipContextMap[shipId] ?: throw IllegalArgumentException("Context for shipId=$shipId not found")
    }

    fun releaseContext(shipId: EntityId) {
        val context = shipContextMap.remove(shipId)
        Log.log("Released context=${context?.contextId} for shipId=$shipId")
    }

    private fun getNextContextId(): ContextId {
        val allContextIds = 0.until(maxContextCount).map { ContextId(it) }.toSet()
        val takenContextIds = shipContextMap.map { it.value.contextId }.toSet()
        return (allContextIds - takenContextIds).firstOrNull() ?: throw IllegalStateException("No new context ids available")
    }
}

fun toRankedTiles(ship: Ship, gameMap: GameMap, tiles: List<Tile>): List<RankedTile> {
    val tileHaliteAvgs = tiles.map { tile ->
        val positions = tile.getPositionsForAnchor(ship.position)
        val haliteSum = positions.sumBy { gameMap.at(it)?.halite ?: 0 }
        Pair(tile, haliteSum.toDouble() / positions.size)
    }.toMutableList()
    tileHaliteAvgs.sortBy { it.second }
    val tileToRankMap = tileHaliteAvgs.mapIndexed { index, pair ->  pair.first to index }.toMap()
    return tiles.map { RankedTile(tileToRankMap[it]!!, it) }
}

fun toState(contextId: ContextId, ship: Ship, shipyard: Shipyard, gameMap: GameMap): State {
    val agentId = AgentId(contextId.id)

    val twoByTwoRankedTiles = toRankedTiles(ship, gameMap, twoByTwoTiles)

    val shipyardDirections = gameMap.getUnsafeMoves(ship.position, shipyard.position)
    val nearestDropOffDirection = if (shipyardDirections.isEmpty()) Direction.STILL else shipyardDirections.first()

    val shipHaliteLevel = HaliteLevel.fromHalite(ship.halite)

    val cellHaliteLevel = HaliteLevel.fromHalite(gameMap.at(ship)?.halite ?: 0)

    return State(agentId, twoByTwoRankedTiles, nearestDropOffDirection, shipHaliteLevel, cellHaliteLevel)
}

fun toCommand(ship: Ship, action: Action): Command {
    return when (action) {
        is Move -> ship.move(action.direction)
    }
}

fun toNextPosition(ship: Ship, action: Action, gameMap: GameMap): Position {
    return when (action) {
        is Move -> gameMap.normalize(ship.position.directionalOffset(action.direction))
    }
}

fun toPossibleActions(ship: Ship, invalidNextPositions: Set<Position>, gameMap: GameMap): Set<Action> {
    val mapCell = gameMap.at(ship.position)
    val cellHalite = mapCell?.halite?.toDouble() ?: 0.0

    return if (ship.halite < cellHalite / 10) {
        setOf(Action.STILL)
    } else {
        allDirections.mapNotNull {
            val nextPosition = gameMap.normalize(ship.position.directionalOffset(it))
            if (invalidNextPositions.contains(nextPosition)) null else Move(it)
        }.toSet()
    }
}

fun getInvalidNextPositions(
        shipId: EntityId,
        shipyard: Shipyard,
        currShipIds: Set<EntityId>,
        currShipPositions: Map<EntityId, Position>,
        nextShipPositions: Map<EntityId, Position>): Set<Position> {
    return currShipIds.mapNotNull {
        if (shipId == it) {
            if (currShipPositions[it] == shipyard.position) {
                shipyard.position
            } else {
                null
            }
        } else {
            nextShipPositions[it] ?: currShipPositions[it]!!
        }
    }.toSet()
}

object MyBot {
    @JvmStatic
    fun main(args: Array<String>) {
        val rngSeed = System.nanoTime()
        val isTraining =  args.isNotEmpty()
        val random = Random(rngSeed)
        val game = Game()

        // At this point "game" variable is populated with initial map data.
        // This is a good place to do computationally expensive start-up pre-processing.
        // As soon as you call "ready" function below, the 2 second per turn timer will start.
        val actionValueFunctionPath = Paths.get("q.txt")
        val actionValueFunctionStore = ActionValueFunctionStore()
        val stateActionValueMap = if (Files.exists(actionValueFunctionPath)) {
            Log.log("Loading agent state from ${actionValueFunctionPath.fileName}")
            actionValueFunctionStore.load(actionValueFunctionPath)
        } else {
            mutableMapOf()
        }
        val contextManager = ContextManager(random, 4, stateActionValueMap)
        var prevHalite = 0

        game.ready("daVinciBot1495")

        Log.log("Successfully created bot! My Player ID is " + game.myId + ". Bot rng seed is " + rngSeed + ".")

        while (true) {
            game.updateFrame()

            val me = game.me
            val shipyard = me.shipyard
            val gameMap = game.gameMap
            val commandQueue = ArrayList<Command>()

            //----------------------------------------------------------------------------------------------------------
            // Start Reinforcement Learning
            //----------------------------------------------------------------------------------------------------------
            val currShipIds = me.ships.keys
            val prevShipIds = contextManager.getShipIds()
            val removedShipIds = prevShipIds - currShipIds
            val addedShipIds = currShipIds - prevShipIds

            removedShipIds.forEach { contextManager.releaseContext(it) }
            addedShipIds.forEach { contextManager.newContext(it) }

            val currShipPositions = me.ships.values.map { it.id to it.position }.toMap()
            val nextShipPositions = mutableMapOf<EntityId, Position>()

            for (shipId in currShipIds) {
                val invalidNextPositions = getInvalidNextPositions(
                        shipId,
                        shipyard,
                        currShipIds,
                        currShipPositions,
                        nextShipPositions)

                val ship = me.ships[shipId]!!
                val context = contextManager.getContext(shipId)
                val agent = context.agent

                val currHalite = me.halite
                val currState = toState(context.contextId, ship, shipyard, gameMap)
                val currPossibleActions = toPossibleActions(ship, invalidNextPositions, gameMap)
                val currAction = agent.chooseAction(currState, currPossibleActions)
                val prevStateAction = context.prevStateAction

                if (prevStateAction != null) {
                    val reward = if (ship.position == shipyard.position) currHalite - prevHalite else 0
                    agent.learn(
                            prevStateAction.state,
                            prevStateAction.action,
                            reward.toDouble(),
                            currState,
                            currAction,
                            currPossibleActions)
                }

                nextShipPositions[shipId] = toNextPosition(ship, currAction, gameMap)
                context.prevStateAction = StateAction(currState, currAction)
                prevHalite = currHalite

                commandQueue.add(toCommand(ship, currAction))
            }

            val hasEnoughHalite = me.halite > 1000
            val anyNextPositionsInShipyard = nextShipPositions.any { it.value == shipyard.position }

            if (hasEnoughHalite && !anyNextPositionsInShipyard && contextManager.hasContextAvailable()) {
                Log.log("Spawning new ship")
                commandQueue.add(shipyard.spawn())
            }

            if (isTraining && Constants.MAX_TURNS == game.turnNumber) {
                Log.log("Saving state to ${actionValueFunctionPath.fileName}")
                actionValueFunctionStore.save(actionValueFunctionPath, stateActionValueMap)
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
