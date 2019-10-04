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
	void setup();
	void update(ofxGameEngine* game);
	void draw(ofxGameEngine* game);

	static MenuState* Instance() {
		return &instance;
	}
};
#endif /* MENU_STATE_H */