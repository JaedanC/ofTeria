#pragma once
#ifndef PLAY_STATE_H
#define PLAY_STATE_H

#include "ofMain.h"

#include "ofxGameStates/ofxGameState.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "Game/WorldSpawn.h"

class PlayState : public ofxGameState
{
private:
	shared_ptr<WorldSpawn> worldSpawn;
protected:
	PlayState()
		: ofxGameState(
			false, // updateTransparent
			false, // drawTransparent
			"PlayState"
		), worldSpawn(make_shared<WorldSpawn>("worldname"))
	{}
public:
	inline weak_ptr<WorldSpawn> getWorldSpawn() { return worldSpawn; }

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