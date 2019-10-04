#include "ofxGameEngine.h"
#include "ofxGameState.h"

ofxGameEngine ofxGameEngine::instance;

namespace ofxGameEngineGlobal {
	int x = 6;
}

void ofxGameEngine::ChangeState(ofxGameState* state)
{
	if (!states.empty()) {
		//states.back()->Cleanup();
		states.pop_back();
	}

	// store and init the new state
	states.push_back(state);
	states.back()->setup();
}

void ofxGameEngine::PushState(ofxGameState* state)
{
	// pause current state
	/*
	if (!states.empty()) {
		states.back()->Pause();
	}
	*/

	// store and init the new state
	cout << "ofxGameEngine: Pushing " + state->getStateName() << endl;
	states.push_back(state);
	states.back()->setup();
}

void ofxGameEngine::PopState()
{
	if (states.size() == 1) {
		cout << "Can't pop the last GameState! Or close game? \n";
		return;
	}
	// cleanup the current state
	if (!states.empty()) {
		//states.back()->Cleanup();
		states.pop_back();
	}

	// resume previous state
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
	if (states.empty()) {
		cout << "ofxGameEngine states is empty. Can't update(). Closing program.\n";
		assert(false);
	}
	
	/* See ofxGameEngine::draw(). */
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
	// Check if states is empty
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
