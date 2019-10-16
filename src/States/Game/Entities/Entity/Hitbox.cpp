#include "Hitbox.h"
#include "ofMain.h"

//----------------------------------------------------------

void Hitbox::set(ofVec2f* _entityLock, ofVec2f& _entityOffset, float _w, float _h)
{
	entityLock = _entityLock;
	entityOffset = _entityOffset;
	w = _w;
	h = _h;
}

void Hitbox::draw()
{
	ofSetColor(0);
	ofNoFill();
	ofDrawRectangle(*entityLock + entityOffset, w, h);
	ofFill();
}