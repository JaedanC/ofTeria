#pragma once
#ifndef PLAY_STATE_H
#define PLAY_STATE_H
#include "ofMain.h"
#include "ofxGameStates/ofxGameState.h"
#include "ofxGameStates/ofxGameEngine.h"

class PlayState : public ofxGameState
{
private:
protected:
	PlayState()
		: ofxGameState(
			false, // updateTransparent
			false, // drawTransparent
			"PlayState"
		) {}
public:
	virtual void setup();
	virtual void exit() {}
	virtual void update(ofxGameEngine* game);
	virtual void draw(ofxGameEngine* game);

	static PlayState* Instance() {
		static PlayState instance;
		return &instance;
	}
};
#endif /* PLAY_STATE_H */