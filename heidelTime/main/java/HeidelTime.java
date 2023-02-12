package heidelTime.main.java;

import de.unihd.dbs.heideltime.standalone.DocumentType;
import de.unihd.dbs.heideltime.standalone.HeidelTimeStandalone;
import de.unihd.dbs.heideltime.standalone.OutputType;
import de.unihd.dbs.uima.annotator.heideltime.resources.Language;

import java.io.File;
import java.io.FileWriter;
import java.util.*;

public class HeidelTime {

    /*
    normalise the publication time of the claims
    input is a string of publication times with where each entry is split by a tab ("\t") and contains first the id and
    second their unnormalised publication time
    output: timex normalisation files in xml format saved in the folder ProcessedDates
     */
    public static void normalisePublicationTimeClaim(String[] args) throws Exception {

        List<String>processedDates = new ArrayList<>();
        int processed = 0;
        File file = new File(args[1]);
        Scanner sc = new Scanner(file);
        HeidelTimeStandalone heideltime = new HeidelTimeStandalone(Language.ENGLISH, DocumentType.NARRATIVES, OutputType.TIMEML, "config.props");
        while(sc.hasNextLine()) {
            String[] line = sc.nextLine().split("\t");
            String process = heideltime.process(line[1]);
            FileWriter myWriter = new FileWriter("ProcessedDates/"+line[0]+".xml");
            myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') +1)+1));
            myWriter.close();
            processed++;
            System.out.println("Processed : " + processed);
        }
    }

    public static HashMap<String,List<String>> listDirectoriesForFolder(final File folder) {
        HashMap<String,List<String>> directories = new HashMap<>();
        for (final File fileEntry : folder.listFiles()) {
            directories.put(fileEntry.getName(),listFilesForDirectory(fileEntry));
        }
        return directories;
    }

    public static List<String> listFilesForDirectory(final File folder) {
        List<String> files = new ArrayList<>();
        for (final File fileEntry : folder.listFiles()) {
            files.add(fileEntry.getName());
        }
        return files;
    }

    /*
    normalise the publication time of the snippets
    input is path of folder containing files of publication times with where each entry is split by a tab ("\t") and contains first the id and
    second their unnormalised publication time
    output: timex normalisation files in xml format saved in the folder processedSnippets
     */
    public static void normalisePublicationTimeSnippets(String[] args) throws Exception {

        List<String>processedDates = new ArrayList<>();
        int processed = 0;
        final File folder= new File("snippetDates");
        HashMap<String,List<String>> directories = listDirectoriesForFolder(folder);
        for(String directory:directories.keySet()){
            for(String snippet: directories.get(directory)){
                File file = new File("snippetDates/"+directory+"/"+snippet);
                Scanner sc = new Scanner(file);
                HeidelTimeStandalone heideltime = new HeidelTimeStandalone(Language.ENGLISH, DocumentType.NARRATIVES, OutputType.TIMEML, "config.props");
                while(sc.hasNextLine()) {
                    String line = sc.nextLine();
                    String process = heideltime.process(line);
                    File dict = new File("processedSnippets/" +directory);
                    dict.mkdir();
                    FileWriter myWriter = new FileWriter("processedSnippets/" +directory+"/"+snippet+".xml");
                    myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') +1)+1));
                    myWriter.close();
                    processed++;
                    System.out.println("Processed : " + processed);
                }
            }

        }
    }

    public static boolean isInteger(String s) {
        return isInteger(s,10);
    }

    public static boolean isInteger(String s, int radix) {
        if(s.isEmpty()) return false;
        for(int i = 0; i < s.length(); i++) {
            if(i == 0 && s.charAt(i) == '-') {
                if(s.length() == 1) return false;
                else continue;
            }
            if(Character.digit(s.charAt(i),radix) < 0) return false;
        }
        return true;
    }

    /*
    extract timexes from the text of claim/evidence
    input is path of file containing the publication date of claim/evidence and path of file containing indices of these claim/evidence
    output: xml files containing timex extraction in timeml format with name claimId+'claim' or number of evidence
            in folder processedTimes
     */
    public static void processTimexesInText(String[] args) throws Exception {
        File ProcessedTimes = new File("processedTimes");
        if not processTimes.exist(){
            ProcessedTimes.mkdir();
        }
        int processed = 0;
        File file = new File(args[1]);
        File indices = new File(args[2]);
        Scanner sc = new Scanner(file);
        Scanner scIndices = new Scanner(indices);
        HeidelTimeStandalone heideltime = new HeidelTimeStandalone(Language.ENGLISH, DocumentType.NEWS, OutputType.TIMEML, "config.props");
        HeidelTimeStandalone heideltimeN = new HeidelTimeStandalone(Language.ENGLISH, DocumentType.NARRATIVES, OutputType.TIMEML, "config.props");
        while(sc.hasNextLine()) {
            String[] indicesSnippets = scIndices.nextLine().split("\t");
            indicesSnippets = Arrays.copyOfRange(indicesSnippets, 1, indicesSnippets.length);
            String[] line = sc.nextLine().split("\t");
            String[] days = line[1].split("Claim -D ")[1].split(" ")[0].split("-");
            if (days.length!=1){
                String[] hours = line[1].split("Claim -D ")[1].split(" ")[1].split(":");
                File text = new File("text" + '/'+line[0]+'/'+"claim");
                Scanner sc2 = new Scanner(text);
                String document = sc2.nextLine().replace("...","");
                    Date time = new Date(Integer.parseInt(days[0]) - 1900, Integer.parseInt(days[1]) - 1, Integer.parseInt(days[2]), Integer.parseInt(hours[0]), Integer.parseInt(hours[1]), Integer.parseInt(hours[2]));
                    String process = heideltime.process(document, time);
                    File dict = new File("processedTimes/" + line[0]);
                    dict.mkdir();
                    FileWriter myWriter = new FileWriter("processedTimes/" + line[0] + '/' + "claim" + ".xml");
                    myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') + 1) + 1));
                    myWriter.close();
                    processed++;
                    System.out.println("Processed : " + processed);
            }
            else{
                File text = new File("text" + '/'+line[0]+'/'+"claim");
                Scanner sc2 = new Scanner(text);
                String document = sc2.nextLine().replace("...","");
                String process = heideltimeN.process(document);
                File dict = new File("processedTimes/" + line[0]);
                dict.mkdir();
                FileWriter myWriter = new FileWriter("processedTimes/" + line[0] + '/' + "claim" + ".xml");
                myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') + 1) + 1));
                myWriter.close();
                processed++;
                System.out.println("Processed : " + processed);
            }
            for(int i=2;i<line.length;i++){
                days = line[i].split(" ")[0].split("-");
                if (days.length!=1){
                    String[] hours = line[i].split(" ")[1].split(":");
                    File text = new File("text" + '/'+line[0]+'/'+Integer.parseInt(indicesSnippets[i-2]));
                    Scanner sc2 = new Scanner(text);
                    String document = sc2.nextLine().replace("...","");
                    Date time = new Date(Integer.parseInt(days[0]) - 1900, Integer.parseInt(days[1]) - 1, Integer.parseInt(days[2]), Integer.parseInt(hours[0]), Integer.parseInt(hours[1]), Integer.parseInt(hours[2]));
                    String process = heideltime.process(document, time);
                    File dict = new File("processedTimes/" + line[0]);
                    dict.mkdir();
                    FileWriter myWriter = new FileWriter("processedTimes/" + line[0] + '/' + Integer.parseInt(indicesSnippets[i-2]) + ".xml");
                    myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') + 1) + 1));
                    myWriter.close();
                    processed++;
                    System.out.println("Processed : " + processed);
                }
                else{
                    if(days[0]!="Exist") {
                        File text = new File("text" + '/'+line[0]+'/'+Integer.parseInt(indicesSnippets[i-2]));
                        Scanner sc2 = new Scanner(text);
                        String document = sc2.nextLine().replace("...", "");
                        String process = heideltimeN.process(document);
                        File dict = new File("processedTimes/" + line[0]);
                        dict.mkdir();
                        FileWriter myWriter = new FileWriter("processedTimes/" + line[0] + '/' + Integer.parseInt(indicesSnippets[i - 2]) + ".xml");
                        myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') + 1) + 1));
                        myWriter.close();
                        processed++;
                        System.out.println("Processed : " + processed);
                    }
                }
            }

        }
    }


}
