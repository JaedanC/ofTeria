#include "ofApp.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ofxIni/ofxIniFile.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "States/MenuState.h"
#include "Keyboard/KeyboardInput.h"

ofxGameEngine* engine;

//--------------------------------------------------------------
void ofApp::setup(){
	ofxIniFile settings("settings.ini");

	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------
	/* Load the games default settings and apply them to the game. If the settings
	do not exist, write the default values to the settings file. */
	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------
	bool fullscreen		= settings.loadSetBool("game", "fullscreen", false);
	bool vsync			= settings.loadSetBool("game", "vsync", false);
	int target_fps		= settings.loadSetInt("game", "target_fps", 2000);
	if (!fullscreen) {
		// Only set the window size if the game is not fullscreen
		int window_width  = settings.loadSetInt("game", "window_width", 800);
		int window_height = settings.loadSetInt("game", "window_height", 600);
		ofSetWindowShape(window_width, window_height);
	}

	// Will still set the target fps if we are in vsync since it might be turned off later.
	ofSetFrameRate(target_fps);
	ofSetFullscreen(fullscreen);
	ofSetVerticalSync(vsync);
	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------
	
	vector<string> aliases = settings.getAllValues("alias", "alias");
	for (string& alias : aliases) {
		string binding = settings.loadSetString("bindings", alias, "");
		KeyboardInput::Instance()->registerAlias(alias, binding[0]);
	}

	// ------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------


	engine = ofxGameEngine::Instance();
	engine->setup();
	engine->PushState(MenuState::Instance());
}

//--------------------------------------------------------------
void ofApp::update(){
	engine->update();
}

//--------------------------------------------------------------
void ofApp::draw(){
	// Background
	ofBackground(ofColor::dimGrey);

	// Debug Values
	ofSetColor(ofColor::white);
	debugPush("FrameRate: " + ofToString(ofGetFrameRate()));
	debugPush("FrameRate: " + ofToString(ofGetFrameRate()));
	debugPush("WindowPosition: " + ofToString(ofVec2f(ofGetWindowPositionX(), ofGetWindowPositionY())));
	debugPush("WindowSize: " + ofToString(ofGetWindowSize()));

	// GameState draw
	engine->draw();

	// Draw the debug info to screen
	debugDraw();

	//Reset inputs
	KeyboardInput::Instance()->resetPollingMaps();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	KeyboardInput::Instance()->keyPressed(key);
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	KeyboardInput::Instance()->keyReleased(key);
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	KeyboardInput::Instance()->mousePressed(x, y, button);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
	KeyboardInput::Instance()->mouseReleased(x, y, button);
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
