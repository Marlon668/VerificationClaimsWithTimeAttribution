import de.unihd.dbs.heideltime.standalone.DocumentType;
import de.unihd.dbs.heideltime.standalone.HeidelTimeStandalone;
import de.unihd.dbs.heideltime.standalone.OutputType;
import de.unihd.dbs.uima.annotator.heideltime.resources.Language;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class processTimexesInText {

    /*
    extract timexes from the text of claim/evidence
    input is path of file containing the publication date of claim/evidence and path of file containing indices of these claim/evidence
    output: xml files containing timex extraction in timeml format with name claimId+'claim' or number of evidence
            in folder processedTimes
    @arg 1 path to data.txt or file that contains publication date of claim and evidence
    @arg 2 path to file containing indices linking publication date to claim/snippet
    @arg 3 path to the folder containing the text of the claim and snippets
    @arg 4 path to the folder to save the timeml annotations of the claim and snippettext
     */
    public static void main(String[] args) throws Exception {
        Files.createDirectories(Paths.get(args[3]));
        int processed = 0;
        File file = new File(args[0]);
        File indices = new File(args[1]);
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
                File text = new File(args[2] + '/'+line[0]+'/'+"claim");
                Scanner sc2 = new Scanner(text);
                String document = sc2.nextLine().replace("...","");
                Date time = new Date(Integer.parseInt(days[0]) - 1900, Integer.parseInt(days[1]) - 1, Integer.parseInt(days[2]), Integer.parseInt(hours[0]), Integer.parseInt(hours[1]), Integer.parseInt(hours[2]));
                String process = heideltime.process(document, time);
                File dict = new File(args[3] + "/" + line[0]);
                dict.mkdir();
                FileWriter myWriter = new FileWriter(args[3] + "/" + line[0] + '/' + "claim" + ".xml");
                myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') + 1) + 1));
                myWriter.close();
                processed++;
                System.out.println("Processed : " + processed);
            }
            else{
                File text = new File(args[2] + '/'+line[0]+'/'+"claim");
                Scanner sc2 = new Scanner(text);
                String document = sc2.nextLine().replace("...","");
                String process = heideltimeN.process(document);
                File dict = new File(args[3] + "/" + line[0]);
                dict.mkdir();
                FileWriter myWriter = new FileWriter(args[3] + "/" + line[0] + '/' + "claim" + ".xml");
                myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') + 1) + 1));
                myWriter.close();
                processed++;
                System.out.println("Processed : " + processed);
            }
            for(int i=2;i<line.length;i++){
                days = line[i].split(" ")[0].split("-");
                if (days.length!=1){
                    String[] hours = line[i].split(" ")[1].split(":");
                    File text = new File(args[2] + '/'+line[0]+'/'+Integer.parseInt(indicesSnippets[i-2]));
                    Scanner sc2 = new Scanner(text);
                    String document = sc2.nextLine().replace("...","");
                    Date time = new Date(Integer.parseInt(days[0]) - 1900, Integer.parseInt(days[1]) - 1, Integer.parseInt(days[2]), Integer.parseInt(hours[0]), Integer.parseInt(hours[1]), Integer.parseInt(hours[2]));
                    String process = heideltime.process(document, time);
                    File dict = new File(args[3] + "/" + line[0]);
                    dict.mkdir();
                    FileWriter myWriter = new FileWriter(args[3]+"/" + line[0] + '/' + Integer.parseInt(indicesSnippets[i-2]) + ".xml");
                    myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') + 1) + 1));
                    myWriter.close();
                    processed++;
                    System.out.println("Processed : " + processed);
                }
                else{
                    if(days[0]!="Exist") {
                        File text = new File(args[2] + '/'+line[0]+'/'+Integer.parseInt(indicesSnippets[i-2]));
                        Scanner sc2 = new Scanner(text);
                        String document = sc2.nextLine().replace("...", "");
                        String process = heideltimeN.process(document);
                        File dict = new File(args[3] + "/" + line[0]);
                        dict.mkdir();
                        FileWriter myWriter = new FileWriter(args[3] + "/" + line[0] + '/' + Integer.parseInt(indicesSnippets[i - 2]) + ".xml");
                        myWriter.write(process.substring(process.indexOf('\n', process.indexOf('\n') + 1) + 1));
                        myWriter.close();
                        processed++;
                        System.out.println("Processed : " + processed);
                    }
                }
            }

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
}