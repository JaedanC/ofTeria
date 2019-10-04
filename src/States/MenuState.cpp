#include "MenuState.h"
#include "ofMain.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "PlayState.h"
MenuState MenuState::instance;

namespace MenuStateGlobal {
	int x = 5;
}

void MenuState::setup() {
	ofxGameEngine::Instance()->ChangeState(PlayState::Instance());
}

void MenuState::update(ofxGameEngine* game) {
	MenuStateGlobal::x++;
}

void MenuState::draw(ofxGameEngine* game) {
	ofSetColor(ofColor::aqua);
	ofDrawCircle(MenuStateGlobal::x, 500, 50);
}