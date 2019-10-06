#pragma once
#ifndef MENU_STATE_H
#define MENU_STATE_H
#include "ofMain.h"
#include "ofxGameStates/ofxGameState.h"
#include "ofxGameStates/ofxGameEngine.h"

class MenuState : public ofxGameState
{
private:
	static MenuState instance;
protected:
	MenuState()
		: ofxGameState(
			false, // updateTransparent
			false, // drawTransparent
			"MenuState"
		) {}
public:
	virtual void setup();
	virtual void update(ofxGameEngine* game);
	virtual void draw(ofxGameEngine* game);
	virtual void exit() {}

	static MenuState* Instance() {
		return &instance;
	}
};
#endif /* MENU_STATE_H */