#pragma once
#ifndef PLAY_STATE_H
#define PLAY_STATE_H

#include "ofMain.h"

#include "ofxGameStates/ofxGameState.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "Game/WorldSpawn.h"

#define queryPlayStateInput(x, type) PlayState::Instance()->queryInput(x, type)

/* This is the main PlayState of the game. fixedUpdate() is calculated
and called from in here. */
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
	/* Elapsed time between fixedUpdate() calls. If this goes above (1 / fixedUpdateRate) then
	the corresponding number of fixedUpdates() are called. */
	float elapsedTime = 0;

	/* This value is used for interpolation. On a fixed update this is zero. When 1/4 of the time through
	to another fixedUpdate() call this number will be 0.25. Typically multiplied with an entities Velocity
	for interpolation. */
	float framePercentage = 0;

	/* How many fixedUpdates() we will aim to hit every second. If this number is above the frameRate of the
	game it will make up the time loss and potentially call this function more than once in an effort to still
	run the game at a constant speed.*/
	int fixedUpdateRate = 60;

	/* Returns a weak pointer to the WorldSpawn instance that we own. */
	inline weak_ptr<WorldSpawn> getWorldSpawn() { return worldSpawn; }

	/* These functions are called from the ofxGameEngine class and it's updateStack etc. */
	virtual void setup();
	virtual void exit() {}
	virtual void update(ofxGameEngine* game);
	virtual void draw(ofxGameEngine* game);

	/* This function is called outselves from update every (1/fixedUpdateRate) milliseconds. */
	virtual void fixedUpdate(ofxGameEngine* game);

	/* Returns the singleTon instance of this class. */
	static PlayState* Instance() {
		static PlayState instance;
		return &instance;
	}
};
#endif /* PLAY_STATE_H */