#include "Hitbox.h"
#include "ofMain.h"
#include "Entity.h"
#include "../../../PlayState.h"

//----------------------------------------------------------

void Hitbox::setOffset(Entity* _entity, ofVec2f& _entityOffset, float _w, float _h)
{
	entity = _entity;
	entityOffset = _entityOffset;
	w = _w;
	h = _h;
}

void Hitbox::draw()
{
	ofSetColor(0);
	ofNoFill();
	ofDrawRectangle(*entity->getWorldPos() + entityOffset + *entity->getInterp(), w, h);
	ofFill();
}