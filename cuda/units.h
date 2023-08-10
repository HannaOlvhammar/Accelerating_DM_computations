#ifndef UNITS_H
#define UNITS_H

namespace units
{
  const double yotta = 1e24;
  const double zetta = 1e21;
  const double exa = 1e18;
  const double peta = 1e15;
  const double tera = 1e12;
  const double giga = 1e9;
  const double mega = 1e6;
  const double kilo = 1e3;
  const double hecto = 1e2;
  const double deca = 10;
  const double deci = 1e-1;
  const double centi = 1e-2;
  const double milli = 1e-3;
  const double micro = 1e-6;
  const double nano = 1e-9;
  const double pico = 1e-12;
  const double femto = 1e-15;
  const double atto = 1e-18;
  const double zepto = 1e-21;
  const double yocto = 1e-24;
  const double eV = 1;
  const double meV = 1e-3;
  const double keV = 1e3;
  const double MeV = 1e6;
  const double GeV = 1e9;
  const double TeV = 1e12;
  const double PeV = 1e15;
  const double Joule = 6.24150913e18 * eV;
  const double Rydberg = 13.605693009 * eV;
  const double cal = 4.184 * Joule;
  const double electronmass = 510.99895 * keV;  // mass of electron in eV
  const double gram = 5.60958884493318e32 * eV;
  const double kg = 1e3 * gram;
  const double tonne = 1e3 * kg;
  const double lbs = 0.453592 * kg;
  const double AMU = 0.9314940954 * GeV;
  const double Earthmass = 5.9724e24 * kg;
  const double Sunmass = 1.98848e30 * kg;
  const double cm = 5.067730214314311e13/GeV;
  const double mm = 0.1 * cm;
  const double meter = 100 * cm;
  const double km = 1e3 * meter;
  const double fm = 1e-15 * meter;
  const double inch = 2.54 * cm;
  const double foot = 12 * inch;
  const double yard = 3 * foot;
  const double mile = 1609.34 * meter;
  const double Angstrom = 1e-10 * meter;
  const double a0 = 5.291772083e-11 * meter;
  const double pc = 3.08567758e16 * meter;
  const double kpc = 1e3 * pc;
  const double Mpc = 1e6 * pc;
  const double AU = 149597870700 * meter;
  const double Earthradius = 6371 * km;
  const double Sunradius = 695510 * km;
  const double barn = 1e-24 * cm*cm;
  const double pb = 1e-36 * cm*cm;
  const double acre = 4046.86 * meter*meter;
  const double hectare = 10000 * meter*meter;
  const double sec = 299792458 * meter;
  const double ms = 1e-3 * sec;
  const double microsec = 1e-6 * sec;
  const double ns = 1e-9 * sec;
  const double minute = 60 * sec;
  const double hr = 3600 * sec;
  const double day = 24 * hr;
  const double week = 7 * day;
  const double yr = 365.25 * day;
  const double Hz = 1/sec;
  const double erg = gram * cm*cm/(sec*sec);
  const double LightYear = 365.25 * day;
  const double Newton=kg * meter/(sec*sec);
  const double dyne = 1e-5 * Newton;
  const double Watt=Joule/sec;
  const double ElementaryCharge=0.30282212;
}


#endif
