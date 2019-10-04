#pragma once
#ifndef OFX_GAME_STATE_H
#define OFX_GAME_STATE_H

#include "ofMain.h"
#include "ofxGameEngine.h"

class ofxGameState
{
protected:
	/*
	Any Variables for each State should be put here and initialised in the
	constructor below.
	*/
	bool drawTransparent;
	bool updateTransparent;
	string stateName;
	ofxGameState(const bool& updateTransparent, const bool& drawTransparent, const string& stateName) :
		updateTransparent(updateTransparent),
		drawTransparent(drawTransparent),
		stateName(stateName)
	{}

public:
	bool getUpdateTransparent() const {
		return updateTransparent;
	}

	bool getDrawTransparent() const {
		return drawTransparent;
	}

	string getStateName() const {
		return stateName;
	}

	virtual void setup() = 0;
	virtual void update(ofxGameEngine* game) = 0;
	virtual void draw(ofxGameEngine* game) = 0;

	void ChangeState(ofxGameEngine* game, ofxGameState* state) {
		game->ChangeState(state);
	}
};

#endif /* OFX_GAME_STATE_H */