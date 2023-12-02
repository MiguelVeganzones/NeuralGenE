CXX = 				g++

DEBUG_CXXFLAGS =  	-fdiagnostics-color=always \
					-fdiagnostics-show-template-tree \
					-fdiagnostics-path-format=inline-events \
					-fdiagnostics-show-caret \
					-ggdb3 \
					-O0 \
					-Wall \
					-Wextra \
					-Wshadow \
					-Wconversion \
					-fsanitize=address \
					-fsanitize=leak \
					-fsanitize=undefined \
					-Werror \
					-pedantic \
					-mavx \
					-fbounds-check \
					-fconcepts-diagnostics-depth=3 \
					-std=c++23

RELEASE_CXXFLAGS =  -fdiagnostics-color=always \
					-fdiagnostics-show-template-tree \
					-fdiagnostics-path-format=inline-events \
					-fdiagnostics-show-caret \
					-ggdb3 \
					-O2 \
					-Wall \
					-Wextra \
					-Wshadow \
					-Wconversion \
					-Werror \
					-pedantic \
					-mavx \
					-fbounds-check \
					-fconcepts-diagnostics-depth=3 \
					-std=c++23
					
FULL_RELEASE_CXXFLAGS =  -fdiagnostics-color=always \
					-fdiagnostics-show-template-tree \
					-fdiagnostics-path-format=inline-events \
					-fdiagnostics-show-caret \
					-O3 \
					-Wall \
					-Wextra \
					-Wshadow \
					-Wconversion \
					-Werror \
					-pedantic \
					-mavx \
					-fconcepts-diagnostics-depth=3 \
					-std=c++23 \
					-fno-exceptions
					
					
NO_WERROR_DEBUG_CXXFLAGS =  -fdiagnostics-color=always \
							-fdiagnostics-show-template-tree \
							-fdiagnostics-path-format=inline-events \
							-fdiagnostics-show-caret \
							-g \
							-Wall \
							-Wextra \
							-Wshadow \
							-Wconversion \
							-fsanitize=address \
							-fsanitize=leak \
							-fsanitize=undefined \
							-pedantic \
							-mavx \
							-fbounds-check \
							-fconcepts-diagnostics-depth=3 \
							-std=c++23 

NO_WERROR_RELEASE_CXXFLAGS =-fdiagnostics-color=always \
							-fdiagnostics-show-template-tree \
							-fdiagnostics-path-format=inline-events \
							-fdiagnostics-show-caret \
							-O3 \
							-Wall \
							-Wextra \
							-Wshadow \
							-Wconversion \
							-pedantic \
							-mavx \
							-fbounds-check \
							-fconcepts-diagnostics-depth=3 \
							-std=c++23 

RELEASE ?= 0
ifeq (${RELEASE}, 1)
    CXXFLAGS = ${RELEASE_CXXFLAGS}
	OUT_DIR = bin/release
else ifeq (${RELEASE}, 2)
    CXXFLAGS = ${FULL_RELEASE_CXXFLAGS}
	OUT_DIR = bin/full_release
else
    CXXFLAGS = ${DEBUG_CXXFLAGS}
	OUT_DIR = bin/debug
endif


UTILITY_DIR = Utility
MCTS_DIR = MonteCarloTreeSearch
C4_DIR = ConnectFour
SNN_DIR = StaticNeuralNet
PLT_DIR = Plotting
NEURAL_MODEL_DIR = NeuralModel
MINIMAX_SEARCH_DIR = MinimaxTreeSearch
EVOLUTION_AGENT_DIR = EvolutionAgent
C4_AGENT_TRAINING_DIR = C4AgentTraining
TOURNAMENT_DIR = TournamentSelection
BENCHMARKING_DIR = Benchmarking

UTILITY_INCL =		
GENERAL_INCL =			-I./$(UTILITY_DIR)
MCTS_INCL =				-I./$(C4_DIR) $(GENERAL_INCL)
C4_INCL =				$(GENERAL_INCL)
SNN_INCL =				$(GENERAL_INCL)
NEURAL_MODEL_INCL = 	-I./$(SNN_DIR) $(SNN_INCL) \
						-I./$(C4_DIR) $(C4_INCL) $(GENERAL_INCL)
PLT_INCL =				-I./$(SNN_DIR) $(SNN_INCL) -I./$(C4_DIR) $(C4_INCL) \
						-I/home/miguelveganzones/Libraries/matplotplusplus-1.1.0-Linux/include \
						-I./$(NEURAL_MODEL_DIR) $(NEURAL_MODEL_INCL) $(GENERAL_INCL)
MINIMAX_SEARCH_INCL = 	-I./$(C4_DIR) $(C4_INCL) $(GENERAL_INCL) \
						-I./$(NEURAL_MODEL_DIR) $(NEURAL_MODEL_INCL) \
						-I./$(SNN_DIR) $(SNN_INCL)
EVOLUTION_AGENT_INCL = 	$(GENERAL_INCL) -I./$(NEURAL_MODEL_DIR) $(NEURAL_MODEL_INCL) #\
						-I/usr/include/python3.10 \
						-I/home/miguelveganzones/.local/lib/python3.10/site-packages/numpy/core/include
C4_AGENT_TRAINING_INCL  = $(GENERAL_INCL) -I./$(C4_DIR) $(C4_INCL) -I./$(EVOLUTION_AGENT_DIR) $(EVOLUTION_AGENT_INCL)
TOURNAMENT_INCL 		= $(GENERAL_INCL) -I./$(C4_DIR) $(C4_INCL) -I./$(EVOLUTION_AGENT_DIR) $(EVOLUTION_AGENT_INCL)
BENCHMARKING_INCL =		$(GENERAL_INCL) -isystem benchmark/include


PLT_LIB = 				-L/home/miguelveganzones/Libraries/matplotplusplus-1.1.0-Linux/lib -l:libmatplot.a \
						-L/home/miguelveganzones/Libraries/matplotplusplus-1.1.0-Linux/lib/Matplot++ -l:libnodesoup.a
MINIMAX_LIB = 			-ltbb
EVOLUTION_AGENT_LIB =	-lpthread -lutil -ldl #-lpython3.10 -Xlinker -export-dynamic
C4_AGENT_TRAINING_LIB = $(MINIMAX_LIB) $(EVOLUTION_AGENT_LIB)
BENCHMARKING_LIB =		-Lbenchmark/build/src -lbenchmark -lpthread

all: mcts_main c4_main static_nn_main utility_main minimax_main neural_model_main evolution_agent_main

#=================================================================================================
mcts_main: $(MCTS_DIR)/$(OUT_DIR)/c4_main.o $(UTILITY_DIR)/${OUT_DIR}/utility_main.o
	
$(MCTS_DIR)/$(OUT_DIR)/c4_main.o: $(MCTS_DIR)/*.cpp $(MCTS_DIR)/*.hpp
	@echo -e Building $@..."\n"
	@mkdir -p $(MCTS_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(MCTS_INCL) $(MCTS_DIR)/mcts_main.cpp -o $@
	@echo -e Built $@ successfully."\n"
#=================================================================================================

#=================================================================================================
c4_main: $(C4_DIR)/$(OUT_DIR)/c4_main.o $(UTILITY_DIR)/${OUT_DIR}/utility_main.o
	
$(C4_DIR)/$(OUT_DIR)/c4_main.o: $(C4_DIR)/*.cpp $(C4_DIR)/*.hpp
	@echo -e Building $@..."\n"
	@mkdir -p $(C4_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(C4_INCL) $(C4_DIR)/c4_main.cpp -o $@
	@echo -e Built $@ successfully."\n"
#=================================================================================================

#=================================================================================================
static_nn_main: $(SNN_DIR)/$(OUT_DIR)/static_nn_main.o $(UTILITY_DIR)/${OUT_DIR}/utility_main.o
	
$(SNN_DIR)/$(OUT_DIR)/static_nn_main.o: $(SNN_DIR)/*.cpp $(SNN_DIR)/*.hpp
	@echo -e Building $@..."\n"
	@mkdir -p $(SNN_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(SNN_INCL) $(SNN_DIR)/static_nn_main.cpp -o $@
	@echo -e Built $@ successfully."\n"
#=================================================================================================

#=================================================================================================
utility_main: $(UTILITY_DIR)/${OUT_DIR}/utility_main.o

$(UTILITY_DIR)/${OUT_DIR}/utility_main.o: $(UTILITY_DIR)/*.cpp $(UTILITY_DIR)/*.hpp
	@echo -e Building $@..."\n"
	@mkdir -p $(UTILITY_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(UTILITY_INCL) $(UTILITY_DIR)/utility_main.cpp -o $@
	@echo -e Built $@ successfully."\n"
#=================================================================================================

minimax_main:
	@echo Building $@..."\n"
	@mkdir -p $(MINIMAX_SEARCH_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(MINIMAX_SEARCH_INCL) ${MINIMAX_LIB} $(MINIMAX_SEARCH_DIR)/$@.cpp -o $(MINIMAX_SEARCH_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"
	
neural_model_main:
	@echo Building $@..."\n"
	@mkdir -p $(NEURAL_MODEL_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(NEURAL_MODEL_INCL) $(NEURAL_MODEL_DIR)/$@.cpp -o $(NEURAL_MODEL_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"
		
plotting_utility_main:
	@echo Building $@..."\n"
	@mkdir -p $(PLT_DIR)/${OUT_DIR}
	$(CXX) $(PLT_DIR)/$@.cpp  $(NO_WERROR_RELEASE_CXXFLAGS) $(PLT_INCL) $(PLT_LIB) -Xlinker --verbose -o $(PLT_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"

evolution_agent_main:
	@echo Building $@..."\n"
	@mkdir -p $(EVOLUTION_AGENT_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(EVOLUTION_AGENT_INCL) $(EVOLUTION_AGENT_LIB) $(EVOLUTION_AGENT_DIR)/$@.cpp -o $(EVOLUTION_AGENT_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"
	
c4_agent_training:
	@echo Building $@..."\n"
	@mkdir -p $(C4_AGENT_TRAINING_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(C4_AGENT_TRAINING_INCL) $(C4_AGENT_TRAINING_LIB) $(C4_AGENT_TRAINING_DIR)/$@.cpp -o $(C4_AGENT_TRAINING_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"

#=================================================================================================
tournament_selection_main: $(TOURNAMENT_DIR)/$(OUT_DIR)/tournament_selection_main.o $(UTILITY_DIR)/${OUT_DIR}/utility_main.o
	
$(TOURNAMENT_DIR)/$(OUT_DIR)/tournament_selection_main.o: $(TOURNAMENT_DIR)/*.cpp $(TOURNAMENT_DIR)/*.hpp
	@echo -e Building $@..."\n"
	@mkdir -p $(TOURNAMENT_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(TOURNAMENT_INCL) $(TOURNAMENT_DIR)/tournament_selection_main.cpp -o $@
	@echo -e Built $@ successfully."\n"
#=================================================================================================

benchmarking_main:
	@echo Building $@..."\n"
	@mkdir -p $(BENCHMARKING_DIR)/${OUT_DIR}
	$(CXX) $(BENCHMARKING_DIR)/$@.cpp $(CXXFLAGS) $(BENCHMARKING_INCL) $(BENCHMARKING_LIB) -o $(BENCHMARKING_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"

#$(MCTS_DIR)/%.o: %.hpp
#	$(CXX) $(CXXFLAGS) $(GENERAL_INCL) $(MCTS_INCL) $< -c $(MCTS_DIR)/bin/$@

done:
	@echo Built $@ successfully."\n"
