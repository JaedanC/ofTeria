#pragma once
#ifndef PLAY_STATE_H
#define PLAY_STATE_H
#include "ofMain.h"
#include "ofxGameStates/ofxGameState.h"
#include "ofxGameStates/ofxGameEngine.h"

class PlayState : public ofxGameState
{
private:
	static PlayState instance;
protected:
	PlayState()
		: ofxGameState(
			false, // updateTransparent
			false, // drawTransparent
			"PlayState"
		) {}
public:
	void setup();
	void update(ofxGameEngine* game);
	void draw(ofxGameEngine* game);

	static PlayState* Instance() {
		return &instance;
	}
};
#endif /* PLAY_STATE_H */