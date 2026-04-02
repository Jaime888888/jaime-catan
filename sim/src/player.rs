use crate::board::{Port, Resource, ResourceBank};
use crate::types::{DevCardHand, PlayerId};

#[derive(Clone, Debug)]
pub struct Player {
    pub id: PlayerId,
    pub resources: ResourceBank,
    pub dev_cards: DevCardHand,
    pub played_knights: u8,
    pub settlements_left: u8,
    pub cities_left: u8,
    pub roads_left: u8,
    pub has_three_to_one_port: bool,
    pub two_to_one_ports: [bool; 5],
}

impl Player {
    pub fn new(id: PlayerId) -> Self {
        Self {
            id,
            resources: ResourceBank([0; 5]),
            dev_cards: DevCardHand::EMPTY,
            played_knights: 0,
            settlements_left: 5,
            cities_left: 4,
            roads_left: 15,
            has_three_to_one_port: false,
            two_to_one_ports: [false; 5],
        }
    }

    pub fn trade_rate(&self, resource: Resource) -> u8 {
        if self.two_to_one_ports[resource as usize] {
            2
        } else if self.has_three_to_one_port {
            3
        } else {
            4
        }
    }

    pub fn update_ports(&mut self, port: Port) {
        match port {
            Port::ThreeToOne => self.has_three_to_one_port = true,
            Port::TwoToOne(r) => self.two_to_one_ports[r as usize] = true,
        }
    }
}
