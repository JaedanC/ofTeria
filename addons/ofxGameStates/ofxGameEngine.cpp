#include "ofxGameEngine.h"
#include "ofxGameState.h"
#include "../src/Keyboard/KeyboardInput.h"

unsigned int ofxGameEngine::StackSize() const
{
	return states.size();
}

ofxGameState* ofxGameEngine::getState() const
{
	return states.back();
}

void ofxGameEngine::ChangeState(ofxGameState* state)
{
	// TODO: Remove
	cout << "ofxGameEngine: Changing current state to " + state->getStateName() << endl;

	// Call the exit function before removing the GameState from the stack
	if (!states.empty()) {
		states.back()->exit();
		states.pop_back();
	}

	// Push the new state and call setup()
	states.push_back(state);
	states.back()->setup();
}

void ofxGameEngine::PushState(ofxGameState* state)
{
	cout << "ofxGameEngine: Pushing " + state->getStateName() << endl;

	// Pause current state ??
	/*
	if (!states.empty()) {
		states.back()->Pause();
	}
	*/

	// Push the new state and call setup()
	states.push_back(state);
	states.back()->setup();
}

void ofxGameEngine::PopState()
{
	// Can't push the last state
	if (states.size() == 1) {
		cout << "Can't pop the last GameState! Or close game? \n";
		return;
	}

	// Call the exit function before removing the GameState from the stack
	if (!states.empty()) {
		states.back()->exit();
		states.pop_back();
	}

	// Resume previous state ??
	/*
	if (!states.empty()) {
		states.back()->Resume();
	}
	*/
}

void ofxGameEngine::setup()
{
}

void ofxGameEngine::update()
{
	/* This section hansles how states recieve inputs passed down from other states. Firstly States all
	stores two unordered_maps:
		1: Stores which Aliases this GameState will let through
		2: Stores which Aliases are not blocked passed down through the Stack
	This code Retrieves these three maps. This is a three step process:
		1: Copy the previousGameState's passMap to our own.
		2: Apply the previousGameState's registeredBlockMap to our passMap.
		3: Perform steps 1 and 2 for every GameState in the Stack starting as the top.
	The State at the top is assumed to recieve all inputs (none are blocked). Everything is done twice as the
	blocking maps are done for Alias' and just raw keys. */
	for (auto i = states.rbegin(); i != states.rend() - 1; ++i) {
		ofxGameState* thisState = *i;
		ofxGameState* nextState = *(i + 1);

		/* Retrieve maps*/
		unordered_map<string, bool>* thisAliasPassesMap = thisState->getAliasPasses();
		unordered_map<string, bool>* nextAliasPassesMap = nextState->getAliasPasses();
		unordered_map<string, KeyboardInputBlockingType>* thisRegisteredAliasBlocksMap = thisState->getRegisteredAliasBlocks();
		
		unordered_map<int, KeyboardInputBlockingType>* thisRegisteredKeyBlocksMap = thisState->getRegisteredKeyBlocks();
		unordered_map<int, bool>* thisKeyPassesMap = thisState->getKeyPasses();
		unordered_map<int, bool>* nextKeyPassesMap = nextState->getKeyPasses();

		/* Copy this state to the next. */
		nextAliasPassesMap->insert(thisAliasPassesMap->begin(), thisAliasPassesMap->end());
		nextKeyPassesMap->insert(thisKeyPassesMap->begin(), thisKeyPassesMap->end());

		/* Update the next state based on the current registered blocks. */
		for (auto& entry : (*thisRegisteredAliasBlocksMap)) {
			// pair<string, KeyboardInputBlockingType>
			// pair<alias, passType>
			if (entry.second == INPUT_BLOCK) {
				(*nextAliasPassesMap)[entry.first] = false;
			} else {
				(*nextAliasPassesMap)[entry.first] = true;
			}
		}

		for (auto& entry : (*thisRegisteredKeyBlocksMap)) {
			// pair<int, KeyboardInputBlockingType>
			// pair<key, passType>
			if (entry.second == INPUT_BLOCK) {
				(*nextKeyPassesMap)[entry.first] = false;
			}
			else {
				(*nextKeyPassesMap)[entry.first] = true;
			}
		}
	}

	// Close program if the stack is empty
	// TODO: Change this behaviour?
	if (states.empty()) {
		cout << "ofxGameEngine states is empty. Can't update(). Closing program.\n";
		assert(false);
	}
	
	/* See ofxGameEngine::draw(). Essentially draws them in reverse order with drawTransparency
	taken into consideration. Many reverse iterators are needed here. */
	vector<ofxGameState*> toUpdate;
	toUpdate.reserve(states.size());

	for (vector<ofxGameState*>::reverse_iterator i = states.rbegin(); i != states.rend(); ++i) {
		ofxGameState* state = *i;
		toUpdate.push_back(state);
		if (!state->getUpdateTransparent()) {
			break;
		}
	}

	for (vector<ofxGameState*>::reverse_iterator i = toUpdate.rbegin(); i != toUpdate.rend(); ++i) {
		ofxGameState* state = *i;
		state->update(this);
	}
}

void ofxGameEngine::draw()
{
	// Close program if the stack is empty
	// TODO: Change this behaviour?
	if (states.empty()) {
		cout << "ofxGameEngine states is empty. Can't draw(). Closing program.\n";
		assert(false);
	}

	/* Create a Vector of States that we are going to draw but assume that
	this vector could be the same size as our states vector. */
	vector<ofxGameState *> toDraw;
	toDraw.reserve(states.size());

	/* Scan the GameStates stack backwards and keep pushing the states that we need to draw
	onto the toDraw vector. If we find a state that is not drawTransparent() we should not go to
	the next element. 
	*/
	for (vector<ofxGameState *>::reverse_iterator i = states.rbegin(); i != states.rend(); ++i) {
		ofxGameState* state = *i;
		toDraw.push_back(state);
		if (!state->getDrawTransparent()) {
			break;
		}
	}

	/* Then, we need to draw these states in reverse to preserve the draw ordering. */
	for (vector<ofxGameState*>::reverse_iterator i = toDraw.rbegin(); i != toDraw.rend(); ++i) {
		ofxGameState* state = *i;
		state->draw(this);
	}

	/*
	Example:

	DrawTransparent State Vector Boolean
	eg.			{F, F, F, T, T, F, T, T, T}
	keep:		{               A, B, C, D}		// Keep the corrent States.

	toDraw<>:	{D, C, B, A}					// This is what the toDraw vector will look like.
	draw:		{A, B, C, D}					// Preserve draw order by drawing toDraw<> backwards.
	*/
}
