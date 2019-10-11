#include "Entity.h"
#include "../EntityController.h"

Entity::Entity(EntityController* entityController)
	: entityController(entityController)
{
	cout << "Constructing Entity\n";
}
