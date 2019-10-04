#include "MenuState.h"
#include "ofMain.h"
MenuState MenuState::instance;

namespace MenuStateGlobal {
	int x = 5;
}

void MenuState::setup() {
}

void MenuState::update(ofxGameEngine* game) {
	MenuStateGlobal::x++;
}

void MenuState::draw(ofxGameEngine* game) {
	ofSetColor(ofColor::aqua);
	ofDrawCircle(MenuStateGlobal::x, 500, 50);
}