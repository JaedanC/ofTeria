#include "MenuState.h"
#include "ofMain.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "PlayState.h"

void MenuState::setup() {
	ofxGameEngine::Instance()->ChangeState(PlayState::Instance());
}

void MenuState::update(ofxGameEngine* game) {
	x++;
}

void MenuState::draw(ofxGameEngine* game) {
	ofSetColor(ofColor::aqua);
	ofDrawCircle(x, 500, 50);
}